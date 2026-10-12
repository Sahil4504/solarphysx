"""Figures for the verification gate (reads results/gate/*.csv) -> results/gate/fig_*.png
   python -m src.experiments.plot_gate"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.dayahead_bench import fit_derate, load
from src.experiments.gate_eval import OUT, PRED, daily_error

INK, MUTED, GRID = "#0B0B0B", "#52514E", "#E6E6E3"
STATUS = {"PASS": "#0CA30C", "FLAG": "#FAB219", "BLOCK": "#D03B3B"}  # status palette: always paired with a text label
S1, S2, S3 = "#2A78D6", "#EB6834", "#1BAF7A"  # categorical slots 1-3
NAME = {"B_physics_residual": "B Physics + ML residual", "A_lgbm_quantile": "A LightGBM quantile",
        "D_solartrans_physics_s0": "D SolarTrans + physics", "C_tft_s0": "C TFT",
        "solartrans_original_s0": "SolarTrans (original inputs)", "pvlib_physical": "pvlib physical",
        "persistence": "Persistence", "smart_persistence": "Smart persistence", "measured (reality)": "Measured (reality)"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C8C8C4")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def checks_figure():
    c = pd.read_csv(OUT / "claims.csv")
    t1 = pd.read_csv(OUT / "t1_violations_pct_days.csv", index_col=0).t1_fail.sort_values()
    mono = pd.read_csv(OUT / "t2_monotonicity_pct_hours.csv", index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

    ax = axs[0]  # claim verdicts per explainer
    order = ["A + SHAP", "B + SHAP", "B + physics explainer"]
    share = pd.crosstab(c.setup, c.verdict, normalize="index").reindex(order)[["PASS", "FLAG", "BLOCK"]] * 100
    n = c.setup.value_counts()
    left = np.zeros(len(order))
    for v in ["PASS", "FLAG", "BLOCK"]:
        ax.barh(range(len(order)), share[v], left=left, color=STATUS[v], height=0.6, edgecolor="white", lw=2)
        for i, (l, w) in enumerate(zip(left, share[v])):
            if w >= 6:
                ax.text(l + w / 2, i, f"{v}\n{w:.0f}%", ha="center", va="center", fontsize=8, color=INK)
        left += share[v].to_numpy()
    ax.set_yticks(range(len(order)), [f"{o}\n({n[o]:,} claims)" for o in order], fontsize=9, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel("% of claims", fontsize=8, color=MUTED)
    ax.set_title("T2 + T3: are the explanation's claims physically true?", fontsize=10, color=INK, loc="left")
    _style(ax)

    ax = axs[1]  # T1
    ax.barh(range(len(t1)), t1, color=S1, height=0.6)
    for i, v in enumerate(t1):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(t1)), [NAME.get(k, k) for k in t1.index], fontsize=8.5, color=INK)
    ax.set_xlim(0, max(t1.max() * 1.25, 5)); ax.set_xlabel("% of test days with a physically impossible output", fontsize=8, color=MUTED)
    ax.set_title("T1: does the forecast break physics?", fontsize=10, color=INK, loc="left")
    _style(ax)

    ax = axs[2]  # T2 model monotonicity
    m = mono.loc[["pvlib_physical", "A_lgbm_quantile", "B_physics_residual"]]
    y = np.arange(len(m))
    ax.barh(y + 0.17, m.clearer_lowers, height=0.32, color=S1, label="clearer sky → LOWER power")
    ax.barh(y - 0.17, m.hotter_raises, height=0.32, color=S2, label="hotter air → HIGHER power")
    for i in range(len(m)):
        ax.text(m.clearer_lowers.iloc[i] + 0.15, i + 0.17, f"{m.clearer_lowers.iloc[i]:.1f}%", va="center", fontsize=8, color=INK)
        ax.text(m.hotter_raises.iloc[i] + 0.15, i - 0.17, f"{m.hotter_raises.iloc[i]:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [NAME[k] for k in m.index], fontsize=8.5, color=INK)
    ax.set_xlim(0, m.max().max() * 1.3); ax.set_xlabel("% of daytime hours responding against physics", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("T2: does the model respond the way physics says?", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.suptitle("Verification gate on the day-ahead test period (Jan–Sep 2023)", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_gate_checks.png", dpi=200); plt.close(fig)


def value_figure():
    df = load()
    te = df[(df.split == "test") & df.daytime]
    d = pd.read_csv(OUT / "days.csv", parse_dates=["day"]).set_index("day")
    x = d[d.setup == "B + physics explainer"]
    err = daily_error(df, "B_physics_residual")
    A = pd.read_parquet(PRED / "pred_A_lgbm_quantile.parquet").loc[te.index]
    j = pd.DataFrame({"pcs": x.pcs, "k": te.k.groupby(te.day).mean(),
                      "narrow": -(A.p90 - A.p10).groupby(te.day).mean()}).join(err.rename("r"), how="inner")
    def curve(order):
        o = order.to_numpy(); n = np.arange(1, len(o) + 1)
        return n / len(o), np.sqrt(np.cumsum(o ** 2) / n)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    for col, lab, cl in [("pcs", "Gate: Physical Consistency Score", S1), ("k", "GFS forecast clearness", S2),
                         ("narrow", "A's P10–P90 interval width", S3)]:
        cov, r = curve(j.sort_values(col, ascending=False, kind="stable").r)
        ax.plot(cov * 100, r, color=cl, lw=2, label=lab)
    cov, r = curve(j.r.sort_values())
    ax.plot(cov * 100, r, color=MUTED, lw=1.2, ls="--", label="Oracle (true error known)")
    ax.axhline(r[-1], color=MUTED, lw=0.8, ls=":")
    ax.text(3, r[-1] + 0.01, f"all days: {r[-1]:.3f} kW", fontsize=8, color=MUTED)
    ax.set_xlabel("Coverage: % of days the operator accepts (most trusted first)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("RMSE of accepted days (kW)", fontsize=8.5, color=MUTED)
    ax.set_xlim(10, 100); ax.set_ylim(0, None)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Risk–coverage, forecaster B (lower curve = better trust ranking)", fontsize=10, color=INK, loc="left")
    _style(ax); ax.grid(axis="y", color=GRID, lw=0.6)

    ax = axs[1]
    g = x.join(err.rename("r"), how="inner").groupby("verdict").r
    rm = g.apply(lambda v: np.sqrt((v ** 2).mean())).reindex(["PASS", "FLAG", "BLOCK"])
    cnt = g.size().reindex(rm.index)
    ax.bar(range(3), rm, color=[STATUS[v] for v in rm.index], width=0.6)
    for i, v in enumerate(rm):
        ax.text(i, v + 0.015, f"{v:.2f} kW\n{cnt.iloc[i]} days", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), rm.index, fontsize=9, color=INK)
    ax.set_ylabel("Forecast RMSE on those days (kW)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0, rm.max() * 1.3)
    ax.set_title("Forecast error by gate verdict (B + physics explainer)", fontsize=10, color=INK, loc="left")
    _style(ax); ax.grid(axis="x", visible=False); ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gate_value.png", dpi=200); plt.close(fig)


def example_figure():
    """One day as the operator would see it: forecast vs measured, and every claim with its verdict."""
    df = load()
    c = pd.read_csv(OUT / "claims.csv", parse_dates=["day"])
    p = c[c.setup == "B + physics explainer"]
    s = c[c.setup == "B + SHAP"]
    # a day where the physics explainer has PASS and BLOCK claims and SHAP also has a BLOCK: shows both failure kinds
    cand = sorted(set(p[p.verdict == "BLOCK"].day) & set(p[p.verdict == "PASS"].day) & set(s[s.verdict == "BLOCK"].day))
    day = cand[len(cand) // 2]
    rows = df[df.day == day]
    B = pd.read_parquet(PRED / "pred_B_physics_residual.parquet").p50.reindex(rows.index)
    fig = plt.figure(figsize=(14, 5.4))
    ax = fig.add_axes([0.05, 0.14, 0.3, 0.72])
    ax.plot(rows.hour, fit_derate(df) * rows.p_cs1, ":", color=MUTED, label="Clear-sky physics")
    ax.plot(rows.hour, B, color=S1, lw=2, label="Forecast B")
    ax.plot(rows.hour, rows.y, "o", color=INK, ms=4, label="Measured")
    ax.set_xlabel("Hour (ACST)", fontsize=8.5, color=MUTED); ax.set_ylabel("kW", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False); _style(ax)
    ax.set_title(f"{day:%d %b %Y}: forecast issued 18:00 the day before", fontsize=10, color=INK, loc="left")
    y = 0.9
    for title, cl in [("SolarPhysX physics explainer → gate", p), ("SHAP narrative → gate", s)]:
        fig.text(0.39, y, title, fontsize=10, color=INK, weight="bold"); y -= 0.055
        for _, r in cl[cl.day == day].iterrows():
            fig.text(0.39, y, r.verdict, fontsize=8.5, color="white", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc=STATUS[r.verdict], ec="none"))
            body = r.text.split(", ", 1)[1]  # keep "window: claim"
            why = f"re-forecast says {r.true_kwh:+.2f} kWh" if r.driver != "ml_correction" else "read from the model"
            fig.text(0.445, y, f"{body}   ({why})", fontsize=8.3, color=INK)
            y -= 0.048
        y -= 0.03
    fig.savefig(OUT / "fig_gate_example.png", dpi=200); plt.close(fig)
    return day


if __name__ == "__main__":
    checks_figure(); value_figure(); print("example day:", example_figure())
