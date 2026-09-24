"""Collects results/e1/*.json into tables and figures.

- Main runs (no tag): mean +/- std across seeds -> e1_summary.csv, e1_summary.png
- Tagged runs (e.g. --tag full60, physics, origin40) -> listed separately
- E1 runs with a per-epoch history -> e1_test_curves.png (test R2 at every epoch;
  monitoring only, never used for model selection)
- Rolling-origin runs -> e1_rolling_origin.png
- E2 feature-set runs -> e2_features.png

Usage (from the project root):
    python -m src.experiments.summarize_e1
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/e1")
PAPER_R2 = {"1": 0.9692, "2": 0.7956, "combined": 0.8967}  # Table 5 of the baseline


def load() -> tuple[pd.DataFrame, list[dict]]:
    rows, raw = [], []
    for f in sorted(OUT.glob("plant*_s*.json")):
        r = json.loads(f.read_text())
        r.setdefault("tag", "")
        r.setdefault("features", "calendar")
        r.setdefault("train_frac", 0.7)
        raw.append(r)
        rows.append({
            "plant": r["plant"], "split": r["split"], "tag": r["tag"], "seed": r["seed"],
            "features": r["features"], "train_frac": r["train_frac"], "clip": r.get("clip", 0.0),
            "leak_%": 100 * r["neighbour_leak_share"],
            "MAE": r["model"]["mae"], "RMSE": r["model"]["rmse"], "R2": r["model"]["r2"],
            "persist_R2": r["persistence"]["r2"],
            "skill_vs_persist": 1 - r["model"]["rmse"] / r["persistence"]["rmse"],
            "R2_day_shuffled": r["day_shuffled"]["r2"] if r["day_shuffled"] else float("nan"),
            "epochs": r["epochs_run"], "best_epoch": r.get("best_epoch"),
        })
    df = pd.DataFrame(rows)
    df["R2_drop_day_shuffled"] = df["R2"] - df["R2_day_shuffled"]
    return df.sort_values(["plant", "split", "tag", "seed"]), raw


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["R2", "MAE", "RMSE", "persist_R2", "skill_vs_persist", "R2_drop_day_shuffled", "leak_%"]
    g = df.groupby(["plant", "split", "tag"])
    agg = g[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    agg["paper_R2"] = agg["plant"].map(PAPER_R2)
    return agg


def plot_main(agg: pd.DataFrame) -> None:
    main = agg[agg["tag"] == ""]
    plants = sorted(main["plant"].unique())

    def get(split, col):
        vals, errs = [], []
        for p in plants:
            row = main[(main.plant == p) & (main.split == split)]
            vals.append(float(row[f"{col}_mean"].iloc[0]) if len(row) else float("nan"))
            e = float(row[f"{col}_std"].iloc[0]) if len(row) else 0.0
            errs.append(0.0 if pd.isna(e) else e)
        return vals, errs

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 0.27

    ax = axes[0]
    paper = [PAPER_R2[p] for p in plants]
    series = [("paper (reported)", paper, [0] * len(plants)),
              ("reproduced, random split", *get("random", "R2")),
              ("reproduced, chronological", *get("chrono", "R2"))]
    for i, (label, vals, errs) in enumerate(series):
        bars = ax.bar([j + (i - 1) * w for j in range(len(plants))], vals, w,
                      yerr=errs, capsize=3, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(range(len(plants)), [f"Plant {p}" for p in plants])
    ax.set(ylabel="Test R² (z-scored)", ylim=(0, 1.1), title="Accuracy under each split protocol")
    ax.legend(fontsize=7, loc="lower left")

    for ax, col, title, ylabel in [
        (axes[1], "skill_vs_persist", "Skill over persistence (1 − RMSE/RMSE_persist)",
         "skill (0 = no better than persistence)"),
        (axes[2], "R2_drop_day_shuffled", "Reliance on the day-of-month index",
         "R² lost when day-of-month is shuffled"),
    ]:
        for i, split in enumerate(["random", "chrono"]):
            vals, errs = get(split, col)
            bars = ax.bar([j + (i - 0.5) * 0.35 for j in range(len(plants))], vals, 0.35,
                          yerr=errs, capsize=3,
                          label="random split" if split == "random" else "chronological split")
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(plants)), [f"Plant {p}" for p in plants])
        ax.set(title=title, ylabel=ylabel)
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT / "e1_summary.png", dpi=200)
    print(f"figure -> {OUT / 'e1_summary.png'}")


def plot_curves(raw: list[dict]) -> None:
    runs = [r for r in raw if r.get("test_r2_history")
            and r["features"] == "calendar" and r["train_frac"] == 0.7]  # E1 runs only
    if not runs:
        return
    plants = sorted({r["plant"] for r in runs})
    fig, axes = plt.subplots(1, len(plants), figsize=(6 * len(plants), 4), squeeze=False)
    for ax, p in zip(axes[0], plants):
        for r in [r for r in runs if r["plant"] == p]:
            label = f"{r['split']}{' ' + r['tag'] if r['tag'] else ''} s{r['seed']}"
            style = "-" if r["split"] == "random" else "--"
            ax.plot(range(1, len(r["test_r2_history"]) + 1), r["test_r2_history"], style, label=label)
        ax.axhline(PAPER_R2[p], color="grey", lw=0.8, ls=":", label="paper R²")
        ax.set(title=f"Plant {p}: test R² at every epoch (monitoring only)",
               xlabel="epoch", ylabel="test R²")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "e1_test_curves.png", dpi=200)
    print(f"figure -> {OUT / 'e1_test_curves.png'}")


def plot_rolling(df: pd.DataFrame) -> None:
    """E1 robustness: skill of the published (calendar) model on each chronological test week.
    Uses the clipped runs only, so every point shares one training protocol."""
    d = df[(df.split == "chrono") & (df.features == "calendar") & (df["clip"] > 0)]
    if d.train_frac.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in d.groupby("plant"):
        m = g.groupby("train_frac")["skill_vs_persist"].mean()
        line = ax.plot(m.index, m.values, "o-", label=f"Plant {p}, chronological")
        rnd = df[(df.plant == p) & (df.split == "random") & (df.tag == "")]["skill_vs_persist"].mean()
        ax.axhline(rnd, color=line[0].get_color(), ls=":", label=f"Plant {p}, random split")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="training fraction (test week moves later →)", ylabel="skill over persistence",
           title="Rolling-origin: skill on each test week (clip 1.0)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "e1_rolling_origin.png", dpi=200)
    print(f"figure -> {OUT / 'e1_rolling_origin.png'}")


def plot_features(df: pd.DataFrame) -> None:
    """E2: same model, same honest split, same (clipped) training, only the input features change."""
    d = df[(df.split == "chrono") & (df.train_frac == 0.7) & (df["clip"] > 0)]
    if d.features.nunique() < 2:
        return
    order = [f for f in ["calendar", "nocal", "physics"] if f in set(d.features)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ylabel in [(axes[0], "R2", "test R²"),
                            (axes[1], "skill_vs_persist", "skill over persistence")]:
        for i, p in enumerate(sorted(d.plant.unique())):
            g = d[d.plant == p].groupby("features")[col].agg(["mean", "std"]).reindex(order)
            bars = ax.bar([j + (i - 0.5) * 0.35 for j in range(len(order))], g["mean"], 0.35,
                          yerr=g["std"].fillna(0), capsize=3, label=f"Plant {p}")
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(order)), order)
        ax.set(ylabel=ylabel, title=f"E2 (chronological, clip 1.0): {ylabel}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "e2_features.png", dpi=200)
    print(f"figure -> {OUT / 'e2_features.png'}")


def paired_e2(df: pd.DataFrame) -> pd.DataFrame:
    """E2 significance: each feature set vs calendar, paired by seed (same split, same clip)."""
    from scipy.stats import ttest_rel
    d = df[(df.split == "chrono") & (df.train_frac == 0.7) & (df["clip"] > 0)]
    rows = []
    for p, g in d.groupby("plant"):
        for f in ["nocal", "physics"]:
            for col in ["skill_vs_persist", "R2"]:
                w = g.pivot_table(index="seed", columns="features", values=col)
                if not {"calendar", f} <= set(w.columns):
                    continue
                w = w[["calendar", f]].dropna()
                diff = w[f] - w["calendar"]
                t, pv = ttest_rel(w[f], w["calendar"]) if len(w) > 1 else (float("nan"),) * 2
                rows.append({"plant": p, "vs_calendar": f, "metric": col, "n_pairs": len(w),
                             "seeds": list(w.index), "diff_mean": diff.mean(), "diff_std": diff.std(),
                             "n_positive": int((diff > 0).sum()), "t": t, "p": pv})
    out = pd.DataFrame(rows)
    if len(out):
        out.round(4).to_csv(OUT / "e2_paired.csv", index=False)
    return out


if __name__ == "__main__":
    df, raw = load()
    pd.set_option("display.width", 250)
    print("=== all runs ===")
    print(df.round(4).to_string(index=False))
    agg = aggregate(df)
    print("\n=== mean (std) across seeds ===")
    show = ["plant", "split", "tag", "n_seeds", "R2_mean", "R2_std", "persist_R2_mean",
            "skill_vs_persist_mean", "skill_vs_persist_std", "R2_drop_day_shuffled_mean"]
    print(agg[show].round(4).to_string(index=False))
    df.round(4).to_csv(OUT / "e1_runs.csv", index=False)
    agg.round(4).to_csv(OUT / "e1_summary.csv", index=False)
    plot_main(agg)
    plot_curves(raw)
    plot_rolling(df)
    plot_features(df)
    print("\n=== E2 paired by seed (feature set - calendar) ===")
    print(paired_e2(df).round(4).to_string(index=False))
