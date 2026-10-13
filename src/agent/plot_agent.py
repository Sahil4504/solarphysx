"""One real test day through the whole framework: forecast anatomy, driver effects, and the agent's explain -> verify -> revise trace.
  python -m src.agent.plot_agent [YYYY-MM-DD]      (needs results/agent/template_log.csv)  -> results/agent/fig_agent_day.png
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent.explainer import FORECASTER, OUT
from src.experiments.gate_eval import setup
from src.experiments.plot_gate import INK, MUTED, S1, S2, S3, STATUS, _style

DRIVER_C = {"cloud": S1, "temperature": S2, "ml_correction": S3}


def main(day="2023-01-07", name="template"):
    df, m, F = setup()
    rows = df[df.day == day]
    d = m["derate"]
    log = pd.read_csv(OUT / f"{name}_log.csv")
    log = log[log.day == day]
    b2 = np.clip(F[FORECASTER](rows), 0, 5.9)
    b = np.clip(F["B_physics_residual"](rows), 0, 5.9)

    nl = len(log) + 2 * log["round"].nunique() + 3  # text lines in the trace
    H = 5.8 + 0.3 * nl  # inches: fixed-size charts on top, trace grows with the number of claims
    fig = plt.figure(figsize=(15, H))
    top, step = 1 - 4.6 / H, 0.3 / H
    ax = fig.add_axes([0.05, top, 0.42, 3.7 / H])  # A: forecast anatomy
    h = rows.hour.to_numpy()
    for x in h[rows.cs_ghi.to_numpy() == 0]:
        ax.axvspan(x - 0.5, x + 0.5, color="#E9E9E6", lw=0)
    series = [(d * rows.p_cs1, ":", MUTED, "clear-sky physics"), (d * rows.p_phys1, "-", S2, "physics + GFS weather"),
              (b, "--", MUTED, "B (before)"), (b2, "-", S1, "B2 forecast")]
    for y, ls, c, lab in series:
        ax.plot(h, y, ls, color=c, lw=2)
    ax.plot(h, rows.y, "o", color=INK, ms=5)
    ends = {"clear-sky physics": (d * rows.p_cs1).max(), "physics + GFS weather": float((d * rows.p_phys1).iloc[-5]),
            "B2 forecast": float(b2[-6]), "measured": float(rows.y.iloc[-4])}
    ax.annotate("clear-sky physics", (h[7], ends["clear-sky physics"]), (h[2], ends["clear-sky physics"] - 0.6), fontsize=8.5, color=MUTED)
    ax.annotate("physics with GFS cloud + temperature", (h[-5], ends["physics + GFS weather"]), (h[-6] - 0.2, ends["physics + GFS weather"] + 0.9),
                fontsize=8.5, color=S2, arrowprops=dict(arrowstyle="-", color=S2, lw=0.6))
    ax.annotate("B2 forecast = physics + ML residual", (h[-6], b2[-6]), (h[-10], b2[-6] - 1.2), fontsize=8.5, color=S1,
                arrowprops=dict(arrowstyle="-", color=S1, lw=0.6))
    ax.annotate("measured", (h[-4], ends["measured"]), (h[-3] - 0.4, ends["measured"] + 0.5), fontsize=8.5, color=INK)
    night = h[(rows.cs_ghi.to_numpy() == 0)]
    if len(night):
        ax.text(night[-1], 0.25, "sun down:\nB2 forced to 0\n(B: dashed)", fontsize=8, color=MUTED, ha="center")
    ax.set_xlabel("Hour (ACST)", fontsize=8.5, color=MUTED); ax.set_ylabel("kW", fontsize=8.5, color=MUTED)
    ax.set_ylim(0, (d * rows.p_cs1).max() * 1.12); _style(ax); ax.grid(axis="y", color="#E6E6E3", lw=0.6)
    ax.set_title(f"1  Forecaster: {pd.Timestamp(day):%d %b %Y}, issued 18:00 the day before", fontsize=10.5, color=INK, loc="left")

    ax = fig.add_axes([0.56, top, 0.41, 3.7 / H])  # B: driver effects (verified, model-true values)
    shown = log[log["round"] == log["round"].max()]
    shown = shown[shown.verdict == "PASS"]
    wins = ["morning", "midday", "afternoon"]
    drivers = ["cloud", "temperature", "ml_correction"]
    for j, dr in enumerate(drivers):
        for i, w in enumerate(wins):
            v = shown[(shown.window == w) & (shown.driver == dr)].delta_kwh
            if len(v):
                y = i * 4 + j
                ax.barh(y, v.iloc[0], color=DRIVER_C[dr], height=0.8)
                ax.text(v.iloc[0] + (0.06 if v.iloc[0] >= 0 else -0.06), y, f"{dr.replace('_', ' ')} {v.iloc[0]:+.2f}",
                        va="center", ha="left" if v.iloc[0] >= 0 else "right", fontsize=8.3, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks([i * 4 + 1 for i in range(3)], ["morning\n06–10", "midday\n10–14", "afternoon\n14–19"], fontsize=9, color=INK)
    ax.invert_yaxis()
    lo, hi = ax.get_xlim(); ax.set_xlim(lo - 1.2, hi + 1.0)
    ax.set_xlabel("Effect on window energy (kWh), as verified by re-running B2", fontsize=8.5, color=MUTED)
    _style(ax)
    ax.set_title("2  What drives it (claims that passed the gate)", fontsize=10.5, color=INK, loc="left")

    # C: agent trace
    y = top - 1.0 / H
    fig.text(0.05, y, "3  Agent: explain → verify (gate) → revise", fontsize=10.5, color=INK)
    y -= 1.3 * step
    for rnd, g in log.groupby("round"):
        head = "Round 0 — first draft (physics-chain estimates)" if rnd == 0 else f"Round {rnd} — after gate feedback"
        fig.text(0.05, y, head, fontsize=9.5, color=INK, weight="bold"); y -= 1.15 * step
        for _, r in g.iterrows():
            fig.text(0.05, y, f" {r.verdict} ", fontsize=8.3, color="white", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc=STATUS.get(r.verdict, MUTED), ec="none"))
            why = "" if r.verdict == "PASS" else (f"   ← gate: re-running B2 gives {r.true_kwh:+.2f} kWh" +
                                                  ("  (wrong direction)" if r.t3 == "BLOCK" else "  (wrong size)"))
            fig.text(0.11, y, r.sentence + why, fontsize=8.3, color=INK if r.verdict == "PASS" else STATUS[r.verdict])
            y -= step
        y -= 0.4 * step
    n0 = (log["round"] == 0).sum(); nf = len(shown)
    fig.text(0.05, y - 0.3 * step, f"Shown to the operator: {nf} verified sentences (from {n0} drafted); anything that could not be "
             "verified is withheld.", fontsize=9, color=INK)
    fig.savefig(OUT / "fig_agent_day.png", dpi=170); plt.close(fig)
    print("saved", OUT / "fig_agent_day.png")


if __name__ == "__main__":
    main(*sys.argv[1:2])
