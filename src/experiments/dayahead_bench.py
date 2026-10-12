"""Day-ahead benchmark on DKASC #38 (issue 18:00 ACST, next-day hourly power; train 2021-01..2022-09, val 2022-Q4, test 2023-01..09).

Baselines: persistence, smart persistence, pvlib physical model.
Candidates: A LightGBM quantile | B physics + LightGBM residual | C Temporal Fusion Transformer (pytorch-forecasting)
            | D SolarTrans-style transformer with physics inputs; plus SolarTrans with its original inputs (calendar, no NWP).
Every model writes results/dayahead/pred_<name>.parquet (index time; p50 [, p10, p90]); --summary scores them.
  python -m src.experiments.dayahead_bench --models all      then      python -m src.experiments.dayahead_bench --summary
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA, OUT = "data/processed/dayahead38.parquet", Path("results/dayahead")
CAP = 5.9
NWP = ["k", "tcc", "t2m", "ghi_gfs", "poa_gfs", "p_phys1", "p_cs1", "zenith"]  # known for day D at issue time
HIST = ["p_lag", "pcs_lag", "kd_lag"]
FEATS = NWP + ["hour"] + HIST
QS = [0.1, 0.5, 0.9]


def load():
    df = pd.read_parquet(DATA)
    df["daytime"] = df.zenith < 85
    return df


def fit_derate(df):
    tr = df[(df.split == "train") & df.daytime].dropna(subset=["y"])
    return float((tr.y * tr.p_phys1).sum() / (tr.p_phys1 ** 2).sum())


def save(name, pred: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    pred.clip(lower=0, upper=CAP).to_parquet(OUT / f"pred_{name}.parquet")
    print("saved", name)


# ---------------- baselines ----------------
def baselines(df):
    save("persistence", df[["p_lag"]].set_axis(["p50"], axis=1))
    save("smart_persistence", (df.kd_lag * df.p_cs1).to_frame("p50"))  # yesterday's clearness x tomorrow's clear-sky power
    save("pvlib_physical", (fit_derate(df) * df.p_phys1).to_frame("p50"))


# ---------------- A, B: LightGBM ----------------
def _lgb(X, y, Xv, yv, **kw):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, **kw)
    return m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])


def lightgbm_models(df, save_preds=True):
    """Fit A (3 quantiles) and B; returns the fitted models (the gate re-runs them on counterfactual inputs)."""
    d = df.dropna(subset=["y"])
    tr, va = d[d.split == "train"], d[d.split == "val"]
    preds, models = {}, {}
    for q in QS:
        m = _lgb(tr[FEATS], tr.y, va[FEATS], va.y, objective="quantile", alpha=q)
        preds[f"p{int(q * 100)}"] = m.predict(df[FEATS])
        models[f"A{int(q * 100)}"] = m
    p = pd.DataFrame(preds, index=df.index)
    p[["p10", "p50", "p90"]] = np.sort(p[["p10", "p50", "p90"]].to_numpy(), axis=1)  # no quantile crossing
    if save_preds:
        save("A_lgbm_quantile", p)

    phys = fit_derate(df) * df.p_phys1
    r = d.y - phys.loc[d.index]
    m = _lgb(tr[FEATS], r.loc[tr.index], va[FEATS], r.loc[va.index], objective="regression")
    if save_preds:
        save("B_physics_residual", (phys + m.predict(df[FEATS])).to_frame("p50"))
    return {**models, "B_resid": m, "derate": fit_derate(df)}


# ---------------- C: TFT ----------------
def tft(df, seed=0, max_epochs=40):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(seed)
    d = df.copy()
    d["time_idx"] = ((d.day - d.day.min()).dt.days * 15 + d.hour - 5).astype(int)
    d["series"] = "38"
    d["y_missing"] = d.y.isna().astype(float)
    # shortcut: ~5% missing targets filled with the physical estimate so the library accepts them; never scored
    d["yf"] = d.y.fillna(fit_derate(df) * d.p_phys1).astype(np.float32)
    d[NWP] = d[NWP].astype(np.float32)
    enc, pred_len = 28, 17  # encoder ends 17:00 on D-1 (issue 18:00); predict D-1 18:00 .. D 19:00, score only day D
    common = dict(time_idx="time_idx", target="yf", group_ids=["series"], max_encoder_length=enc,
                  max_prediction_length=pred_len, min_prediction_length=pred_len, min_encoder_length=enc,
                  time_varying_known_reals=NWP + ["hour"], time_varying_unknown_reals=["yf", "y_missing"],
                  allow_missing_timesteps=True, add_relative_time_idx=True, scalers={}, target_normalizer=None)
    tr_end = d.loc[d.split == "train", "time_idx"].max()
    va_start, va_end = d.loc[d.split == "val", "time_idx"].agg(["min", "max"])
    te_start = d.loc[d.split == "test", "time_idx"].min()
    train = TimeSeriesDataSet(d[d.time_idx <= tr_end], **common)
    starts_at_18 = lambda idx: (idx.time_idx_first_prediction % 15) == 13
    val = TimeSeriesDataSet.from_dataset(train, d[d.time_idx <= va_end], min_prediction_idx=va_start - 2).filter(starts_at_18)
    test = TimeSeriesDataSet.from_dataset(train, d, min_prediction_idx=te_start - 2).filter(starts_at_18)

    model = TemporalFusionTransformer.from_dataset(train, hidden_size=32, attention_head_size=4, dropout=0.1,
                                                   hidden_continuous_size=16, learning_rate=1e-3,
                                                   loss=QuantileLoss(QS), log_interval=-1)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", gradient_clip_val=0.1, enable_progress_bar=False,
                         logger=False, enable_checkpointing=False,
                         callbacks=[EarlyStopping("val_loss", patience=6)], enable_model_summary=False)
    trainer.fit(model, train.to_dataloader(train=True, batch_size=64, num_workers=0),
                val.to_dataloader(train=False, batch_size=256, num_workers=0))
    out = model.predict(test.to_dataloader(train=False, batch_size=256), mode="quantiles", return_index=True)
    q = out.output.cpu().numpy()  # (n, 17, 3)
    first = out.index.time_idx.to_numpy()
    rows = []
    t2time = pd.Series(d.index, index=d.time_idx)
    for i, t0 in enumerate(first):
        for j in range(2, pred_len):  # drop D-1 18:00, 19:00
            if t0 + j in t2time.index:
                rows.append((t2time[t0 + j], *q[i, j]))
    p = pd.DataFrame(rows, columns=["time", "p10", "p50", "p90"]).set_index("time")
    save(f"C_tft_s{seed}", p)


# ---------------- D (and SolarTrans original inputs): SolarTrans transformer ----------------
def _day_tensors(df, enc_cols, dec_cols):
    """Encoder: D-2 05-19 + D-1 05-17 (28 steps, measured up to issue). Decoder: day D, 15 steps of known inputs."""
    days = sorted(df.day.unique())
    by = {k: g for k, g in df.groupby("day")}
    E, Dd, Y, M, T, S = [], [], [], [], [], []
    for d in days:
        p1, p2 = d - pd.Timedelta(days=1), d - pd.Timedelta(days=2)
        if p1 not in by or p2 not in by or len(by[d]) != 15:
            continue
        e = pd.concat([by[p2], by[p1][by[p1].hour < 18]])
        E.append(e[enc_cols].to_numpy(np.float32)); Dd.append(by[d][dec_cols].to_numpy(np.float32))
        Y.append(np.nan_to_num(by[d].y.to_numpy(np.float32) / CAP)); M.append(by[d].y.notna().to_numpy())
        T.append(by[d].index); S.append(by[d].split.iloc[0])
    return np.stack(E), np.stack(Dd), np.stack(Y), np.stack(M), T, np.array(S)


def solartrans(df, variant="D_solartrans_physics", seed=0, epochs=150, patience=15):
    import torch
    from src.models.solartrans import SolarTrans

    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = df.copy()
    d["y_obs"] = (d.y / CAP).fillna(0); d["y_missing"] = d.y.isna().astype(float)
    doy = d.index.dayofyear
    d["hs"], d["hc"] = np.sin(2 * np.pi * d.hour / 24), np.cos(2 * np.pi * d.hour / 24)
    d["ds"], d["dc"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if variant.startswith("D"):  # physics + NWP, no calendar
        enc_cols, dec_cols = ["y_obs", "y_missing", "p_cs1", "zenith"], NWP
    else:  # as published: measured history + calendar, no weather forecast
        enc_cols, dec_cols = ["y_obs", "y_missing", "hs", "hc", "ds", "dc"], ["hs", "hc", "ds", "dc"]
    E, Dd, Y, M, T, S = _day_tensors(d, enc_cols, dec_cols)
    tr = S == "train"
    mu_e, sd_e = E[tr].mean((0, 1)), E[tr].std((0, 1)) + 1e-6
    mu_d, sd_d = Dd[tr].mean((0, 1)), Dd[tr].std((0, 1)) + 1e-6
    E, Dd = (E - mu_e) / sd_e, (Dd - mu_d) / sd_d
    t = lambda a: torch.tensor(a, device=dev)
    model = SolarTrans(n_enc=E.shape[2], n_dec=Dd.shape[2], enc_len=E.shape[1], dec_len=15).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    def loss_on(idx, train):
        model.train(train)
        out = model(t(E[idx]), t(Dd[idx]))
        m = t(M[idx]).float()
        return ((out - t(Y[idx])) ** 2 * m).sum() / m.sum()
    tri, vai = np.where(tr)[0], np.where(S == "val")[0]
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        for b in np.array_split(np.random.permutation(tri), max(1, len(tri) // 32)):
            opt.zero_grad(); l = loss_on(b, True); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            v = float(loss_on(vai, False))
        if ep % 10 == 0:
            print(f"  {variant} s{seed} ep {ep} val {v:.4f}", flush=True)
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        out = model(t(E), t(Dd)).cpu().numpy() * CAP
    p = pd.DataFrame({"p50": out.ravel()}, index=pd.DatetimeIndex(np.concatenate(T), name="time"))
    print(f"{variant} seed {seed} ({dev}): stopped epoch {ep}, best val mse {best:.4f}")
    save(f"{variant}_s{seed}", p)


# ---------------- scoring ----------------
def summary(df, n_boot=1000):
    te = df[(df.split == "test") & df.daytime & df.y.notna()]
    preds = {f.stem[5:]: pd.read_parquet(f) for f in sorted(OUT.glob("pred_*.parquet"))}
    common = te.index
    for p in preds.values():
        common = common.intersection(p.dropna(subset=["p50"]).index)
    y = te.y.loc[common]
    days = te.day.loc[common].to_numpy()
    uniq = np.unique(days); rng = np.random.default_rng(0)
    boot_idx = [np.concatenate([np.where(days == u)[0] for u in rng.choice(uniq, len(uniq))]) for _ in range(n_boot)]
    e_pers = (preds["persistence"].p50.loc[common] - y).to_numpy()
    rows = []
    for name, p in preds.items():
        e = (p.p50.loc[common] - y).to_numpy()
        rmse = np.sqrt((e ** 2).mean())
        sk = [1 - np.sqrt((e[i] ** 2).mean()) / np.sqrt((e_pers[i] ** 2).mean()) for i in boot_idx]
        r = dict(model=name, rmse_kW=rmse, nrmse_pct=100 * rmse / CAP, mae_kW=np.abs(e).mean(), bias_kW=e.mean(),
                 skill_vs_persistence=1 - rmse / np.sqrt((e_pers ** 2).mean()),
                 skill_ci_lo=np.percentile(sk, 2.5), skill_ci_hi=np.percentile(sk, 97.5))
        if "p10" in p:
            lo, hi = p.p10.loc[common], p.p90.loc[common]
            r["coverage80_pct"] = 100 * ((y >= lo) & (y <= hi)).mean()
            r["pinball_kW"] = np.mean([np.mean(np.maximum(q * (y - p[f"p{int(q*100)}"].loc[common]),
                                                          (q - 1) * (y - p[f"p{int(q*100)}"].loc[common]))) for q in QS])
        rows.append(r)
    m = pd.DataFrame(rows)
    # paired day-bootstrap: RMSE(model) - RMSE(best single run); CI excluding 0 => genuinely different from the best
    errs = {name: (p.p50.loc[common] - y).to_numpy() for name, p in preds.items()}
    best = m.sort_values("rmse_kW").model.iloc[0]
    for i, name in enumerate(m.model):
        d = [np.sqrt((errs[name][b] ** 2).mean()) - np.sqrt((errs[best][b] ** 2).mean()) for b in boot_idx]
        m.loc[i, ["d_rmse_vs_best_lo", "d_rmse_vs_best_hi"]] = np.percentile(d, [2.5, 97.5])
    # seed aggregation for the neural models
    m["family"] = m.model.str.replace(r"_s\d+$", "", regex=True)
    agg = m.groupby("family").agg(n=("model", "size"), **{c: (c, "mean") for c in m.columns if c not in ("model", "family")})
    sd = m.groupby("family").skill_vs_persistence.std().rename("skill_sd")
    agg = agg.join(sd).sort_values("rmse_kW")
    m.to_csv(OUT / "metrics_runs.csv", index=False); agg.to_csv(OUT / "metrics.csv")
    print(f"test: {len(common)} daytime hours, {len(uniq)} days")
    print(agg.round(3).to_string())
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="results folder (e.g. results/dayahead_gpu to keep CPU runs)")
    a = ap.parse_args()
    OUT = Path(a.out)  # module global, read by save() and summary()
    df = load()
    ms = set(a.models)
    if ms & {"all", "baselines"}: baselines(df)
    if ms & {"all", "lgbm"}: lightgbm_models(df)
    for s in a.seeds:
        if ms & {"all", "solartrans"}:
            solartrans(df, "D_solartrans_physics", s); solartrans(df, "solartrans_original", s)
        if ms & {"all", "tft"}: tft(df, s)
    if a.summary: summary(df)
