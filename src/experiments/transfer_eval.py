"""Zero-shot transfer: models trained on DKASC #38 only, tested on unseen US sites (Feb 2022 - Jan 2023).

Sites: henderson (PVDAQ 1423, hot desert like Alice Springs), golden (PVDAQ 50, semi-arid mountain, 1996 modules).
Variants: nameplate capacity (strict zero-shot) and _cal (one effective-capacity number from pre-test history).
Power is in DKASC-equivalent kW, so skill and nRMSE (% of capacity) compare directly with the DKASC benchmark.
Writes results/transfer/<site>[_cal]/pred_*.parquet, metrics.csv, seasons.csv.
  python -m src.data.build_dayahead henderson [--calibrate]   (tables first)   then   python -m src.experiments.transfer_eval   (figure only: ... transfer_eval plot; gate: ... transfer_eval gate)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import src.data.build_dayahead as BD
import src.experiments.dayahead_bench as bench
from src.experiments.gate_eval import setup

OUT = Path("results/transfer")
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def forecasts(us: pd.DataFrame, F: dict, derate: float) -> dict:
    """Same forecasters as on DKASC, applied unchanged (no refit) to the US table."""
    p = {"persistence": us.p_lag, "smart_persistence": us.kd_lag * us.p_cs1}
    p.update({k: pd.Series(f(us), index=us.index) for k, f in F.items()})
    return {k: v.to_frame("p50") for k, v in p.items()}


def seasons(us: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Skill vs persistence and nRMSE per season on the hours every model covers (as in bench.summary)."""
    te = us[us.daytime & us.y.notna()]
    preds = {f.stem[5:]: pd.read_parquet(f).p50 for f in sorted(folder.glob("pred_*.parquet"))}
    common = te.index
    for v in preds.values():
        common = common.intersection(v.dropna().index)
    y, s = te.y.loc[common], te.loc[common].index.month.map(SEASON)
    rows = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        m = (s == season)
        r_pers = np.sqrt(((preds["persistence"].loc[common][m] - y[m]) ** 2).mean())
        for name, v in preds.items():
            rmse = np.sqrt(((v.loc[common][m] - y[m]) ** 2).mean())
            rows.append(dict(season=season, model=name, days=te.loc[common][m].day.nunique(),
                             skill=1 - rmse / r_pers, nrmse_pct=100 * rmse / BD.CAP))
    t = pd.DataFrame(rows).pivot(index="model", columns="season", values="skill")[["DJF", "MAM", "JJA", "SON"]]
    pd.DataFrame(rows).to_csv(folder / "seasons.csv", index=False)
    return t




def figure():
    """Skill vs persistence by season at each US site: physics only, B2 strict zero-shot, B2 + capacity calibration."""
    import matplotlib.pyplot as plt
    from src.experiments.plot_gate import INK, MUTED, S1, S2, S3, _style
    home = pd.read_csv("results/dayahead_gpu/metrics.csv", index_col=0).loc["B2_physics_constrained", "skill_vs_persistence"]
    bars = [("pvlib physics only (calibrated)", "_cal", "pvlib_physical", S2),
            ("B2, strict zero-shot (nameplate)", "", "B2_physics_constrained", S1),
            ("B2 + capacity calibration", "_cal", "B2_physics_constrained", S3)]
    groups = ["Year", "DJF", "MAM", "JJA", "SON"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, site, title in zip(axs, ["henderson", "golden"], ["Henderson NV (hot desert, like Alice Springs)",
                                                              "Golden CO (semi-arid mountain, 1996 modules)"]):
        for j, (lab, var, model, c) in enumerate(bars):
            m = pd.read_csv(OUT / f"{site}{var}" / "metrics.csv", index_col=0).loc[model]
            s = pd.read_csv(OUT / f"{site}{var}" / "seasons.csv").query("model == @model").set_index("season").skill
            vals = [m.skill_vs_persistence, *s.reindex(groups[1:])]
            x = np.arange(len(groups)) + (j - 1) * 0.26
            ax.bar(x, vals, width=0.24, color=c, label=lab)
            ax.errorbar(x[0], vals[0], yerr=[[vals[0] - m.skill_ci_lo], [m.skill_ci_hi - vals[0]]], color=INK, lw=1, capsize=3)
            for xi, v in zip(x, vals):
                ax.text(xi, v + (0.012 if v >= 0 else -0.04), f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.axhline(home, color=MUTED, ls="--", lw=1)
        ax.text(4.45, home + 0.012, f"B2 on DKASC (home site) {home:.2f}", ha="right", fontsize=8, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(groups)), ["Whole year\n(95% CI)", "Dec–Feb\nwinter", "Mar–May\nspring", "Jun–Aug\nsummer", "Sep–Nov\nautumn"],
                      fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left"); _style(ax); ax.grid(axis="x", visible=False); ax.grid(axis="y", color="#E6E6E3", lw=0.6)
    axs[0].set_ylabel("skill vs 24-h persistence", fontsize=8.5, color=MUTED)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Zero-shot transfer: trained on Alice Springs only, tested Feb 2022 – Jan 2023 on unseen US sites", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "fig_transfer_seasons.png", dpi=180); plt.close(fig)
    print("saved", OUT / "fig_transfer_seasons.png")


def gate_us(df, m, F, site, variant="_cal"):
    """Verification gate on a US site, everything calibrated on DKASC (T1 bounds, SHAP background, 'average day');
    counterfactual re-forecasts use the US site's physics (BD.use_site)."""
    import src.experiments.gate_eval as GE
    from src.gate import gate as G
    BD.use_site(site)
    us = pd.read_parquet(BD.SITES[site]["out"].replace(".parquet", f"{variant}.parquet"))
    us["daytime"] = us.zenith < 85
    out = OUT / f"{site}{variant}" / "gate"
    out.mkdir(parents=True, exist_ok=True)
    tr = df[df.split == "train"]
    # v1.1: T1 bounds from the site's own pre-test history (same commissioning data as the capacity calibration);
    # DKASC-calibrated bounds flagged real US output on 13% / 35% of days (absolute clear-sky margin, altitude ignored)
    b = G.T1Bounds.calibrate(BD.history_rows(site, BD.effective_kwp(site) if variant == "_cal" else None))
    print(f"{site}{variant} T1 bounds (site history): {b}")
    t1 = {"measured (reality)": G.t1_output(us.y, us, b).mean()}
    t1.update({k: G.t1_output(pd.Series(np.clip(f(us), 0, BD.CAP), index=us.index), us, b).mean() for k, f in F.items()})
    t1 = (100 * pd.DataFrame(t1).T).round(1)
    t1.to_csv(out / "t1_violations_pct_days.csv")
    phi = GE.shap_phi(tr, us, m)
    by_day, sets = GE.stage_claims(us, m, F, phi)
    claims, days, mono = GE.stage_verify(tr, us, F, by_day, sets, b, out)
    # value: forecast error on days by gate verdict (B2 + physics explainer)
    p = pd.read_parquet(OUT / f"{site}{variant}" / "pred_B2_physics_constrained.parquet").p50
    te = us[us.daytime & us.y.notna()]
    err = ((p.loc[te.index] - te.y) ** 2).groupby(te.day).mean() ** 0.5
    d = days[days.setup == "B2 + physics explainer"].join(err.rename("rmse"), how="inner")
    value = d.groupby("verdict").rmse.agg(days="size", rmse_kW=lambda v: np.sqrt((v ** 2).mean()))
    value.to_csv(out / "value_by_verdict.csv")
    print(f"\n===== gate on {site}{variant} =====")
    print("T1 (% days):\n", t1.t1_fail.to_string())
    print("T2 (% daytime hours):\n", mono.to_string())
    print("claims:\n", pd.crosstab(claims.setup, claims.verdict, normalize="index").round(3).to_string())
    print("days (B2 + physics):", days[days.setup == "B2 + physics explainer"].verdict.value_counts(normalize=True).round(3).to_dict())
    print("B2 error by day verdict:\n", value.round(3).to_string())


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["plot"]:
        figure(); sys.exit()
    df, m, F = setup()  # DKASC models, fitted on DKASC train only (asserts they reproduce the saved DKASC forecasts)
    if sys.argv[1:2] == ["gate"]:  # python -m src.experiments.transfer_eval gate   (after the main run)
        for site in ["henderson", "golden"]:
            gate_us(df, m, F, site)
        sys.exit()
    for site in ["henderson", "golden"]:
        for variant in ["", "_cal"]:
            path = Path(BD.SITES[site]["out"].replace(".parquet", f"{variant}.parquet"))
            if not path.exists():
                print("skip (no table):", path); continue
            BD.use_site(site)  # physics for any counterfactual / mask uses the US geometry
            us = pd.read_parquet(path)
            us["daytime"] = us.zenith < 85
            bench.OUT = OUT / f"{site}{variant}"
            for name, p in forecasts(us, F, m["derate"]).items():
                bench.save(name, p)
            print(f"\n===== {site}{variant}: zero-shot, trained on DKASC only =====")
            bench.summary(us)
            print("skill vs persistence by season:\n", seasons(us, bench.OUT).round(3).to_string())
    figure()
