"""E1 - Does random splitting of overlapping windows inflate SolarTrans's accuracy?
Also runs E2 (--features) and rolling-origin evaluation (--train-frac).

Trains the reimplemented SolarTrans with the paper's hyperparameters under one split protocol
and reports test MAE / RMSE / R2 in z-scored units (the units of the paper's Table 5).

Also reports:
- persistence: repeat the last observed DC value over the horizon (no-skill reference)
- day_shuffled: test accuracy after shuffling the day-of-month feature between test windows.
  A large drop means the model leans on the day index (W2).
- neighbour_leak_share: share of test windows with a near-duplicate window in training.

Usage (from the project root):
    python -m src.experiments.e1_leakage --plant 1 --split random
    python -m src.experiments.e1_leakage --plant 1 --split chrono
    python -m src.experiments.e1_leakage --plant 1 --split chrono --features physics
    python -m src.experiments.e1_leakage --plant 1 --split chrono --train-frac 0.4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.baseline_windows import (
    FEATURE_SETS, build_windows, chrono_split_idx, neighbour_leak_share,
    prepare_frame, random_split_idx, zscore_per_inverter,
)
from src.models.solartrans import SolarTrans

# ponytail: the eval-mode "fast path" of nn.TransformerEncoder can return nan on some GPUs even
# when training is fine. Same maths without it; slightly slower eval.
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

PLANTS = {"1": [1], "2": [2], "combined": [1, 2]}
OUT = Path("results/e1")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, T: torch.Tensor, bs: int = 2048) -> np.ndarray:
    model.eval()
    return torch.cat([model(X[i:i + bs], T[i:i + bs]).cpu() for i in range(0, len(X), bs)]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", choices=list(PLANTS), default="1")
    ap.add_argument("--split", choices=["random", "chrono"], default="random")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="debug only: cap batches per epoch")
    ap.add_argument("--tag", default="", help="label for variant runs, e.g. 'full60'")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="calendar", help="E2 feature set")
    ap.add_argument("--train-frac", type=float, default=0.7, help="chrono only: <0.7 moves the test week earlier")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip; 0 = off (published protocol). Use 1.0 if a run diverges to nan")
    ap.add_argument("--skip-existing", action="store_true", help="skip if this run's results file exists (resume a batch)")
    args = ap.parse_args()
    # auto-label variants so each run gets its own results file; defaults keep the E1 names
    parts = [args.features if args.features != "calendar" else "",
             f"origin{round(args.train_frac * 100)}" if args.train_frac != 0.7 else "", args.tag,
             "clip" if args.clip else ""]
    args.tag = "_".join(x for x in parts if x)
    out_file = OUT / f"plant{args.plant}_{args.split}{'_' + args.tag if args.tag else ''}_s{args.seed}.json"
    if args.skip_existing and out_file.exists():
        print(f"skip (exists): {out_file}")
        return
    enc, dec = FEATURE_SETS[args.features]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data (baseline preprocessing, only the split / features differ) ----
    df = prepare_frame(PLANTS[args.plant])
    w = build_windows(df, zscore_per_inverter(df, enc), enc, dec)
    n = len(w["Y"])
    if args.split == "random":
        tr, va, te = random_split_idx(n, seed=args.seed)
    else:
        tr, va, te = chrono_split_idx(w["t_start"], w["t_end"], (args.train_frac, 0.1, 0.2))
    leak = neighbour_leak_share(tr, te, n)
    print(f"plant={args.plant} split={args.split} features={args.features} "
          f"train_frac={args.train_frac} device={dev}")
    print(f"records={len(df)} windows={n} train={len(tr)} val={len(va)} test={len(te)}")
    print(f"test windows with a near-duplicate in train: {100 * leak:.1f}%")

    def gpu(idx, key):
        return torch.from_numpy(w[key][idx]).to(dev)

    Xtr, Ttr, Ytr = gpu(tr, "X"), gpu(tr, "T"), gpu(tr, "Y")
    Xva, Tva, Yva = gpu(va, "X"), gpu(va, "T"), w["Y"][va]
    Xte, Tte, yte = gpu(te, "X"), gpu(te, "T"), w["Y"][te]

    # ---- training (paper: AdamW 5e-4, wd 1e-3, batch 64, MSE, <=60 epochs, early stopping) ----
    model = SolarTrans(n_enc=len(enc), n_dec=len(dec)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    history, test_r2_history, train_loss_history = [], [], []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Ytr), device=dev)
        run_loss = torch.zeros((), device=dev)
        for nb, i in enumerate(range(0, len(perm), 64), start=1):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b], Ttr[b]), Ytr[b])
            loss.backward()
            run_loss += loss.detach()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            if args.max_batches and nb >= args.max_batches:
                break
        train_loss = float(run_loss) / nb
        val_mse = float(((predict(model, Xva, Tva) - Yva) ** 2).mean())
        if not np.isfinite(train_loss):  # stop now instead of burning 60 epochs
            raise SystemExit(f"TRAINING diverged (train loss nan) at epoch {ep}; nothing saved. "
                             f"Re-run with --clip 0.5")
        if not np.isfinite(val_mse):
            raise SystemExit(f"EVALUATION produced nan at epoch {ep} although training loss is "
                             f"{train_loss:.4f}; nothing saved. Report this to Claude.")
        train_loss_history.append(train_loss)
        test_r2 = metrics(yte, predict(model, Xte, Tte))["r2"]  # monitoring only, never used to select
        history.append(val_mse)
        test_r2_history.append(test_r2)
        print(f"epoch {ep:3d} | train {train_loss:.4f} | val MSE {val_mse:.4f} | test R2 {test_r2:.4f} | {time.time() - t0:.0f}s")
        if val_mse < best - 1e-5:
            best, bad, best_ep = val_mse, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)

    # ---- evaluation ----
    res = {"model": metrics(yte, predict(model, Xte, Tte))}

    last = w["X"][te][:, -1, 0:1]  # last observed (z-scored) DC power
    res["persistence"] = metrics(yte, np.repeat(last, yte.shape[1], axis=1))

    res["day_shuffled"] = None  # only meaningful when the day index is an input
    if "day" in enc:
        g = torch.Generator().manual_seed(args.seed)
        shuf = torch.randperm(len(te), generator=g).to(dev)
        Xs, Ts = Xte.clone(), Tte.clone()
        Xs[:, :, enc.index("day")] = Xte[shuf, :, enc.index("day")]
        Ts[:, :, dec.index("day")] = Tte[shuf, :, dec.index("day")]
        res["day_shuffled"] = metrics(yte, predict(model, Xs, Ts))

    summary = {
        "plant": args.plant, "split": args.split, "seed": args.seed, "tag": args.tag,
        "features": args.features, "train_frac": args.train_frac,
        "test_period": [str(w["t_start"][te].min()), str(w["t_end"][te].max())],
        "records": len(df), "windows": n,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "neighbour_leak_share": leak,
        "epochs_run": len(history), "best_epoch": best_ep, "best_val_mse": best, "clip": args.clip,
        "val_mse_history": history, "test_r2_history": test_r2_history,
        "train_loss_history": train_loss_history,
        "units": "z-scored per inverter (as paper Table 5)",
        **res,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))

    print("\n              MAE     RMSE    R2")
    for k in [k for k in ["model", "persistence", "day_shuffled"] if res[k]]:
        m = res[k]
        print(f"{k:13s} {m['mae']:.4f}  {m['rmse']:.4f}  {m['r2']:.4f}")
    print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
