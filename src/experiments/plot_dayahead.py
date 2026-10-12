"""Figures for the day-ahead benchmark: skill bars, example days, framework diagram -> results/dayahead/*.png
   python -m src.experiments.plot_dayahead"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.experiments.dayahead_bench import OUT, load

DARK, AMBER, TEAL, GREY = "#1B2A3A", "#F2A541", "#17808E", "#9AA5B1"  # same palette as the meeting decks
LABEL = {"persistence": "Persistence", "smart_persistence": "Smart persistence", "pvlib_physical": "pvlib physical",
         "solartrans_original": "SolarTrans (original inputs)", "A_lgbm_quantile": "A  LightGBM quantile",
         "B_physics_residual": "B  Physics + ML residual", "C_tft": "C  Temporal Fusion Transformer",
         "D_solartrans_physics": "D  SolarTrans + physics inputs"}
BASE = {"persistence", "smart_persistence", "pvlib_physical", "solartrans_original"}


def skill_bars():
    m = pd.read_csv(OUT / "metrics.csv").sort_values("skill_vs_persistence")
    runs = pd.read_csv(OUT / "metrics_runs.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(m))
    col = [GREY if f in BASE else TEAL for f in m.family]
    ax.barh(y, m.skill_vs_persistence, color=col,
            xerr=[m.skill_vs_persistence - m.skill_ci_lo, m.skill_ci_hi - m.skill_vs_persistence],
            error_kw=dict(ecolor=DARK, lw=1, capsize=3))
    for i, f in enumerate(m.family):  # individual seeds as dots
        s = runs[runs.model.str.replace(r"_s\d+$", "", regex=True) == f].skill_vs_persistence
        if len(s) > 1:
            ax.scatter(s, [i] * len(s), color=AMBER, s=14, zorder=3)
        ax.text(max(m.skill_ci_hi.max(), 0) + 0.02, i, f"nRMSE {m.nrmse_pct.iloc[i]:.1f}%", va="center", fontsize=8, color=DARK)
    ax.set_yticks(y, [LABEL.get(f, f) for f in m.family], fontsize=9)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_xlim(min(-0.05, m.skill_ci_lo.min() - 0.02), m.skill_ci_hi.max() + 0.17)
    ax.set_xlabel("Skill vs persistence (1 - RMSE/RMSE$_{persistence}$), 95% day-bootstrap CI; dots = seeds")
    ax.set_title("Day-ahead (issued 18:00, next day hourly), DKASC #38, test Jan-Sep 2023", fontsize=10, color=DARK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_skill.png", dpi=200); plt.close(fig)


def example_days(best):
    df = load()
    te = df[df.split == "test"]
    dk = te.y.groupby(te.day).sum() / te.p_cs1.groupby(te.day).sum()
    dk = dk[te.y.notna().groupby(te.day).all()]
    picks = {"Clear": dk.idxmax(), "Partly cloudy": (dk - 0.65).abs().idxmin(), "Overcast": dk.idxmin()}
    P = {k: pd.read_parquet(OUT / f"pred_{k}.parquet") for k in ["persistence", "pvlib_physical", best]}
    band = best if "p10" in P[best] else "A_lgbm_quantile"
    if band not in P:
        P[band] = pd.read_parquet(OUT / f"pred_{band}.parquet")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (name, d) in zip(axs, picks.items()):
        g = te[te.day == d]; h = g.hour
        ax.fill_between(h, P[band].p10.reindex(g.index), P[band].p90.reindex(g.index), color=TEAL, alpha=0.18,
                        label=f"80% interval ({LABEL.get(band.rsplit('_s', 1)[0], band)[:2].strip()})")
        ax.plot(h, g.p_cs1 * 0.92, ":", color=GREY, label="Clear-sky physics")
        ax.plot(h, P["persistence"].p50.reindex(g.index), color=GREY, lw=1.2, label="Persistence")
        ax.plot(h, P["pvlib_physical"].p50.reindex(g.index), "--", color=AMBER, lw=1.5, label="pvlib physical")
        ax.plot(h, P[best].p50.reindex(g.index), color=TEAL, lw=2, label=LABEL.get(best.rsplit("_s", 1)[0], best))
        ax.plot(h, g.y, "o", color=DARK, ms=3.5, label="Measured")
        ax.set_title(f"{name}: {d:%d %b %Y}", fontsize=10, color=DARK); ax.set_xlabel("Hour (ACST)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("AC power (kW)")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_example_days.png", dpi=200); plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(12, 5.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.5, 5.2)
    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=c, ec=DARK, lw=1))
        tc = "white" if c in (DARK, TEAL) else DARK
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold", color=tc)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7.6, color=tc, linespacing=1.35)
    def arrow(a, b, txt="", dx=0.0, dy=0.0):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))
        if txt:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, txt, ha="center", fontsize=7, color=DARK, style="italic")
    L = "#EEF2F5"
    box(0.1, 2.9, 2.2, 1.9, "1  Inputs @ 18:00", "NWP: NOAA GFS 0.5°\n(irradiance, cloud, T2m)\nMeasured PV up to issue\nDKASC #38 + Kaggle", L)
    box(0.1, 0.4, 2.2, 1.9, "2  Physics layer (pvlib)", "Sun position, clear-sky\nFitted tilt/azimuth\nTransposition → POA\nCell temperature → P$_{phys}$", L)
    box(2.9, 1.6, 2.4, 2.2, "3  Forecaster", "Best candidate (A-D)\nHorizon 10-24 h (GUI)\nP10 / P50 / P90\nleak-free chronological\nevaluation", TEAL)
    box(5.9, 2.9, 2.6, 1.9, "4  Explainer agent", "LLM + tools: attribution,\nGFS, clear-sky,\ncounterfactual re-forecast\n→ structured claims", AMBER)
    box(5.9, 0.4, 2.6, 1.9, "5  Verification gate", "T1 output bounds\nT2 attribution plausibility\nT3 claim truth vs physics\n→ Physical Consistency Score", DARK)
    box(9.1, 1.6, 2.7, 2.2, "6  Operator view", "Forecast + verified\nexplanations\nPASS   FLAG   BLOCK\nrisk-coverage reporting", L)
    arrow((1.2, 2.9), (1.2, 2.3)); arrow((2.3, 3.8), (2.9, 3.2)); arrow((2.3, 1.4), (2.9, 2.2), "")
    arrow((5.3, 3.2), (5.9, 3.7), "forecast", 0.0, 0.45); arrow((7.2, 2.9), (7.2, 2.3), "claims", 0.38, -0.05)
    ax.annotate("", (6.5, 2.9), (6.5, 2.3), arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.3, connectionstyle="arc3,rad=0"))
    ax.text(6.35, 2.6, "revise\n(reasons)", ha="right", va="center", fontsize=7, color=AMBER, style="italic")
    arrow((8.5, 1.4), (9.1, 2.2), "verdicts", 0.0, -0.55)
    ax.annotate("", (6.9, 0.4), (1.2, 0.4), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1, connectionstyle="arc3,rad=0.2"))
    ax.text(4.0, -0.35, "same physics is the gate's reference", ha="center", fontsize=7, color=GREY, style="italic")
    ax.text(6, 5.05, "SolarPhysX: physics-grounded forecasting with verified explanations", ha="center", fontsize=12,
            weight="bold", color=DARK)
    fig.savefig(OUT / "fig_framework.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def metrics_panels(folder=OUT):
    """All metrics for all models side by side (one panel per metric, never two scales on one axis)."""
    from pathlib import Path
    folder = Path(folder)
    df = load()
    te = df[(df.split == "test") & df.daytime & df.y.notna()]
    P = {f.stem[5:]: pd.read_parquet(f).p50 for f in folder.glob("pred_*.parquet")}
    idx = te.index
    for p in P.values():
        idx = idx.intersection(p.dropna().index)
    y, day = te.y.loc[idx], te.day.loc[idx]
    ss = ((y - y.mean()) ** 2).sum()
    rmse_day = lambda p: ((p.loc[idx] - y) ** 2).groupby(day).mean() ** 0.5
    pers_day = rmse_day(P["persistence"])
    rows = []
    for k, p in P.items():
        e = p.loc[idx] - y
        rows.append(dict(family=k.rsplit("_s", 1)[0] if k[-2:-1] == "s" and k[-1].isdigit() else k,
                         r2=1 - (e ** 2).sum() / ss, rmse=np.sqrt((e ** 2).mean()), mae=e.abs().mean(),
                         days=100 * (rmse_day(p) < pers_day).mean()))
    m = pd.DataFrame(rows).groupby("family").mean()
    m = m.join(pd.read_csv(folder / "metrics.csv").set_index("family")[["skill_vs_persistence", "skill_ci_lo", "skill_ci_hi"]])
    m = m.sort_values("rmse", ascending=False)  # best (lowest RMSE) at the top
    e = 0.92 * df.p_cs1.loc[idx] - y  # weather-blind clear-sky "forecast": R2 you get for free from the sun's path
    r2_free = 1 - (e ** 2).sum() / ss

    blue, grey, ink, muted = "#2A78D6", "#B4BCC5", "#0B0B0B", "#52514E"
    col = [grey if f in BASE else blue for f in m.index]
    yy = np.arange(len(m))
    panels = [("skill_vs_persistence", "Skill vs persistence\n(higher = better)", "{:.3f}"),
              ("rmse", "RMSE, kW\n(lower = better)", "{:.3f}"),
              ("mae", "MAE, kW\n(lower = better)", "{:.3f}"),
              ("r2", "R²\n(higher = better)", "{:.3f}"),
              ("days", "% of days beating persistence\n(higher = better)", "{:.0f}%")]
    fig, axs = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    for ax, (c, title, fmt) in zip(axs, panels):
        v = m[c].to_numpy()
        ax.barh(yy, v, height=0.62, color=col)
        lo = min(0, v.min())
        hi = v.max() * (1.32 if c != "skill_vs_persistence" else 1.0)
        if c == "skill_vs_persistence":
            ax.errorbar(v, yy, xerr=[v - m.skill_ci_lo, m.skill_ci_hi - v], fmt="none", ecolor=muted, lw=0.9, capsize=2)
            hi = m.skill_ci_hi.max() * 1.25
        if c == "r2":
            lo, hi = 0.5, 1.0
            ax.axvline(r2_free, color=muted, lw=1, ls="--")
            ax.text(r2_free, -0.95, f"clear sky, no weather = {r2_free:.2f}", fontsize=7.5, color=muted, ha="center")
            ax.set_ylim(-1.3, len(m) - 0.5)
        for i, val in enumerate(v):
            ref = (c == "days" and m.index[i] == "persistence") or (c == "skill_vs_persistence" and m.index[i] == "persistence")
            x = (m.skill_ci_hi.iloc[i] if c == "skill_vs_persistence" else val)
            ax.text(max(x, lo) + (hi - lo) * 0.02, i, "reference" if ref else fmt.format(val), va="center", fontsize=8, color=ink)
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=9.5, color=ink, loc="left")
        ax.tick_params(axis="x", labelsize=8, colors=muted)
        ax.grid(axis="x", color="#E6E6E3", lw=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#C8C8C4")
    names = [LABEL.get(f, f) + ("   ★ chosen" if f == "B_physics_residual" else "") for f in m.index]
    axs[0].set_yticks(yy, names, fontsize=9, color=ink)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=blue), Patch(color=grey)], ["Candidate models", "Baselines / published reference"],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Day-ahead PV forecast, DKASC #38 — all models, all metrics (test Jan–Sep 2023, {len(idx):,} daytime hours, "
                 "neural models = mean of 3 seeds)", fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(folder / "fig_all_metrics.png", dpi=200); plt.close(fig)
    m.round(4).to_csv(folder / "metrics_all.csv")
    print(m.round(3).to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # e.g. python -m src.experiments.plot_dayahead results/dayahead_gpu  -> metrics figure for that run
        metrics_panels(sys.argv[1]); sys.exit()
    m = pd.read_csv(OUT / "metrics.csv")
    best = m[~m.family.isin(BASE)].sort_values("rmse_kW").family.iloc[0]
    best_run = sorted(OUT.glob(f"pred_{best}*.parquet"))[0].stem[5:]
    skill_bars(); example_days(best_run); framework()
    print("best:", best, "->", best_run)
