"""Gate robustness (Phase 1): does the gate flag the RIGHT claims, independent of our exact settings and physics model?

  bases      for every test day, window and physical driver (cloud vs clear sky, temperature vs 25 C): the true effect of
             that driver on the forecaster B2 (counterfactual re-forecast, as T3) and the physics chain's sign (as T2).
             Everything below is scored from these with the gate's own functions, so no counterfactual is recomputed.
  planted    1.1  plant known errors (sign flip, wrong size, wrong window, wrong cause) and near-correct claims (+-10%);
             detection rate = share of wrong claims not PASSed; false-alarm rate = share of near-correct claims not PASSed.
  sweep      1.2  vary T3 tolerances and the materiality threshold: planted detection/false alarms and the real-claim
             conclusions (physics explainer vs SHAP, per site) across the grid.
  reference  1.3  re-verify the real claims with an independent physics chain (DISC split, Perez transposition, Faiman
             cell temperature) and check the temperature coefficient the chain assumes against measured data.
Outputs: results/robustness/.     python -m src.experiments.gate_robustness [bases|planted|sweep|reference|figure|all]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.build_dayahead as BD
from src.gate import gate as G

OUT = Path("results/robustness")
SITES = ["dkasc38", "henderson", "golden", "elb"]
DRIVERS = {"cloud": "clear_sky", "temperature": "t25"}
FORECASTER = "B2_physics_constrained"


def site_rows(site, df):
    """Test rows of a site with its physics active (US / ELB: capacity-calibrated table, as in transfer_eval.gate_us)."""
    BD.use_site(site)
    if site == "dkasc38":
        return df[df.split == "test"]
    return pd.read_parquet(BD.SITES[site]["out"].replace(".parquet", "_cal.parquet"))


# ---------------- bases ----------------
def bases(df, F, sites=SITES):
    ref = G.references(df[df.split == "train"])
    f, out = F[FORECASTER], []
    for site in sites:
        te = site_rows(site, df)
        for day, rows in te.groupby("day"):
            for w in G.WINDOWS:
                if G.window_rows(rows, w).empty:
                    continue
                for drv, rf in DRIVERS.items():
                    c = G.Claim(day, w, drv, rf, 1.0, "planted", "B2")
                    _, true = G.t3_truth(c, rows, f, ref)
                    s = G.sign_prior(c, rows, ref)
                    out.append(dict(site=site, day=day, window=w, driver=drv, true=true, sign=np.nan if s is None else s))
        print(site, "bases:", sum(o["site"] == site for o in out))
    BD.use_site("dkasc38")
    b = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    b.to_csv(OUT / "bases.csv", index=False)
    return b


# ---------------- scoring (the gate's own rules, from stored truth) ----------------
def verdict(driver, delta, true, sign, tol_abs=G.TOL_ABS, tol_rel=G.TOL_REL):
    c = G.Claim(None, "midday", driver, DRIVERS[driver], delta, "planted", "B2")
    v2 = G.t2_plausibility(c, None, None, s=None if np.isnan(sign) else sign)  # rows/ref unused when the sign is given
    v3 = G.t3_verdict(delta, true, tol_abs, tol_rel)
    return max(v2, v3, key=G.RANK.get), v2, v3


def plant(b, seed=0):
    """Wrong and near-correct claims built from each material true effect (|true| >= materiality)."""
    rng = np.random.default_rng(seed)
    b = b[b.true.abs() >= G.MATERIAL_KWH].copy()
    key = b.set_index(["site", "day", "window", "driver"])
    nxt = dict(zip(G.WINDOWS, list(G.WINDOWS)[1:] + list(G.WINDOWS)[:1]))
    other = {"cloud": "temperature", "temperature": "cloud"}
    rows = []
    for r in b.itertuples():
        own = (r.driver, r.true, r.sign)
        variants = {"correct (exact)": (*own[:1], round(r.true, 2)),
                    "correct (±10%)": (r.driver, round(r.true * (1 + rng.uniform(-0.1, 0.1)), 2)),
                    "sign flipped": (r.driver, -r.true),
                    "size ×0.5": (r.driver, 0.5 * r.true), "size ×1.5": (r.driver, 1.5 * r.true),
                    "size ×2": (r.driver, 2 * r.true), "size ×3": (r.driver, 3 * r.true)}
        for name, (drv, delta) in variants.items():
            rows.append(dict(site=r.site, kind=name, claimed_driver=drv, delta=delta, true=r.true, sign=r.sign))
        # wrong window: this window's effect reported for the next window (judged against that window's truth)
        k = (r.site, r.day, nxt[r.window], r.driver)
        if k in key.index:
            t = key.loc[k]
            rows.append(dict(site=r.site, kind="wrong window", claimed_driver=r.driver, delta=r.true, true=t.true, sign=t.sign))
        # wrong cause: this driver's effect attributed to the other driver (judged against the other driver's truth)
        k = (r.site, r.day, r.window, other[r.driver])
        if k in key.index:
            t = key.loc[k]
            rows.append(dict(site=r.site, kind="wrong cause", claimed_driver=other[r.driver], delta=r.true, true=t.true, sign=t.sign))
    p = pd.DataFrame(rows)
    p = p[p.delta.abs() >= G.MATERIAL_KWH]  # claims below materiality are never made
    # a 'wrong' claim that happens to be within tolerance of the truth is not an error; keep it labelled, report both
    p["really_wrong"] = (np.abs(p.delta - p.true) > np.maximum(G.TOL_ABS, G.TOL_REL * p.true.abs())) | (np.sign(p.delta) != np.sign(p.true))
    return p


def score(p, tol_abs=G.TOL_ABS, tol_rel=G.TOL_REL):
    v = [verdict(d, x, t, s, tol_abs, tol_rel) for d, x, t, s in zip(p.claimed_driver, p.delta, p.true, p.sign)]
    return p.assign(verdict=[a for a, _, _ in v], t2=[b for _, b, _ in v], t3=[c for _, _, c in v])


def planted_table(s):
    s = s.assign(caught=s.verdict != "PASS", blocked=s.verdict == "BLOCK")
    wrong = s[~s.kind.str.startswith("correct")]
    t = s.groupby(["kind", "site"]).caught.mean().unstack()[SITES]
    t["all sites"] = s.groupby("kind").caught.mean()
    t["n"] = s.groupby("kind").size()
    order = ["correct (exact)", "correct (±10%)", "sign flipped", "size ×0.5", "size ×1.5", "size ×2", "size ×3", "wrong window", "wrong cause"]
    t = t.reindex(order)
    # implementation check, not a result: 'really wrong' uses the gate's own tolerance, so every such claim must be caught
    assert wrong[wrong.really_wrong].caught.all(), "a planted error beyond the tolerance passed"
    missed = wrong[~wrong.caught]
    print(f"missed planted errors: {len(missed)}, all within tolerance of the true effect "
          f"(median |claimed - true| {(missed.delta - missed.true).abs().median():.2f} kWh)")
    return t


def planted(b):
    s = score(plant(b))
    s.to_csv(OUT / "planted_claims.csv", index=False)
    t = planted_table(s)
    t.to_csv(OUT / "planted_detection.csv")
    print("\n=== 1.1 planted errors: share NOT passed (correct rows = false alarms) ===\n", (100 * t.drop(columns="n")).round(1).assign(n=t.n).to_string())
    cor = s[s.kind.str.startswith("correct")]
    fa = cor[cor.verdict != "PASS"]
    print("\nfalse alarms on correct claims by reason:", fa.groupby(["kind", "t2", "t3"]).size().to_string())
    # size curve: detection vs claimed / true ratio (sign flip = -1)
    base = b[b.true.abs() >= G.MATERIAL_KWH]
    curve = []
    for r_ in np.round(np.arange(-1.0, 3.01, 0.05), 2):
        d = base.true * r_
        keep = d.abs() >= G.MATERIAL_KWH
        v = [verdict(dr, x, t, sg)[0] for dr, x, t, sg in zip(base.driver[keep], d[keep], base.true[keep], base.sign[keep])]
        curve.append(dict(ratio=r_, caught=np.mean([x != "PASS" for x in v]), n=int(keep.sum())))
    pd.DataFrame(curve).to_csv(OUT / "planted_size_curve.csv", index=False)
    return s, t


# ---------------- 1.2 sweep ----------------
TOL_ABS_GRID, TOL_REL_GRID = [0.1, 0.15, 0.2, 0.3, 0.4], [0.1, 0.15, 0.25, 0.35, 0.5]
REAL = {"dkasc38": "results/gate/claims.csv", "henderson": "results/transfer/henderson_cal/gate/claims.csv",
        "golden": "results/transfer/golden_cal/gate/claims.csv", "elb": "results/transfer/elb_cal/gate/claims.csv"}


def real_pass(c, tol_abs, tol_rel, material=G.MATERIAL_KWH):
    """Re-score saved real claims (T2 verdict kept; T3 recomputed from the saved true effect) at other settings."""
    c = c[c.delta_kwh.abs() >= material]
    t3 = [G.t3_verdict(d, t, tol_abs, tol_rel) if dr != "ml_correction" else "PASS" for d, t, dr in zip(c.delta_kwh, c.true_kwh, c.driver)]
    v = [max(a, b, key=G.RANK.get) for a, b in zip(c.t2, t3)]
    return pd.Series(np.array(v) == "PASS", index=c.index).groupby(c.setup).mean()


def sweep(b):
    p = plant(b)
    rows = []
    for ta in TOL_ABS_GRID:
        for tr in TOL_REL_GRID:
            s = score(p, ta, tr)
            caught = (s.verdict != "PASS").groupby(s.kind).mean()
            rows.append(dict(tol_abs=ta, tol_rel=tr, false_alarm=caught["correct (±10%)"],
                             **{k: caught[k] for k in ["sign flipped", "size ×0.5", "size ×1.5", "size ×2", "wrong window", "wrong cause"]}))
    roc = pd.DataFrame(rows)
    roc.to_csv(OUT / "sweep_planted.csv", index=False)
    print("\n=== 1.2 planted claims caught (%) across tolerances (default 0.20 kWh, 25%) ===")
    print((100 * roc.set_index(["tol_abs", "tol_rel"])).round(1).to_string())
    # real claims: does 'physics explainer passes more than SHAP' hold at every setting, at every site?
    out = []
    for site, path in REAL.items():
        c = pd.read_csv(path)
        for ta in TOL_ABS_GRID:
            for tr in TOL_REL_GRID:
                for mat in [0.05, 0.1, 0.2]:
                    pr = real_pass(c, ta, tr, mat)
                    out.append(dict(site=site, tol_abs=ta, tol_rel=tr, material=mat, **pr.to_dict()))
    real = pd.DataFrame(out)
    real.to_csv(OUT / "sweep_real.csv", index=False)
    real["gap_B2"] = real["B2 + physics explainer"] - real["B2 + SHAP"]
    print("\nB2: physics-explainer PASS minus SHAP PASS (pts) over 75 settings per site: min / median / max")
    print((100 * real.groupby("site").gap_B2.agg(["min", "median", "max"])).round(1).to_string())
    print("ranking physics > SHAP holds for B2 at", f"{(real.gap_B2 > 0).mean():.0%}", "of site x setting combinations")
    return roc, real


# ---------------- 1.3 independent physics reference ----------------
def pv_unit_alt(index, ghi, t_air, wind=1.0):
    """Independent chain with different published sub-models: DISC split, Perez transposition, Faiman cell temperature."""
    import pvlib
    ghi, t_air = np.asarray(ghi, float), np.asarray(t_air, float)
    sp = BD.LOC.get_solarposition(index)
    sp.index = index
    disc = pvlib.irradiance.disc(pd.Series(ghi, index=index), sp.zenith, index)
    dni = disc["dni"].fillna(0).clip(lower=0)
    dhi = (ghi - dni * np.cos(np.radians(sp.zenith)).clip(lower=0)).clip(lower=0)
    extra = pvlib.irradiance.get_extra_radiation(index)
    am = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith)
    poa = pvlib.irradiance.get_total_irradiance(BD.TILT, BD.AZ, sp.apparent_zenith, sp.azimuth, dni, ghi, dhi,
                                                dni_extra=extra, airmass=am, model="perez")["poa_global"].fillna(0).clip(lower=0)
    tc = pvlib.temperature.faiman(poa, t_air, wind)
    return poa, (BD.CAP * poa / 1000 * (1 + BD.GAMMA * (tc - 25))).clip(lower=0), sp.apparent_zenith


def reference(df, F):
    """Re-verify the saved real B2 claims with the gate's physics swapped for the independent chain; agreement of verdicts."""
    ref = G.references(df[df.split == "train"])
    f = F[FORECASTER]
    orig, rows_out = BD.pv_unit, []
    for site, path in REAL.items():
        c = pd.read_csv(path, parse_dates=["day"])
        c = c[c.setup == "B2 + physics explainer"]
        te = site_rows(site, df)
        BD.pv_unit = pv_unit_alt
        try:
            by_day = {k: G._physics(g.copy()) for k, g in te.groupby("day")}  # actual inputs through the alternative chain too
            for r in c.itertuples():
                cl = G.Claim(r.day, r.window, r.driver, r.reference, r.delta_kwh, r.explainer, r.model)
                v2 = G.t2_plausibility(cl, by_day[r.day], ref)
                v3, true = G.t3_truth(cl, by_day[r.day], f, ref)
                rows_out.append(dict(site=site, driver=r.driver, orig=r.verdict, alt=max(v2, v3, key=G.RANK.get),
                                     true_orig=r.true_kwh, true_alt=true))
        finally:
            BD.pv_unit = orig
        print(site, "re-verified", len(c))
    BD.use_site("dkasc38")
    a = pd.DataFrame(rows_out)
    a.to_csv(OUT / "reference_alt_chain.csv", index=False)
    phys = a[a.driver != "ml_correction"]
    print("\n=== 1.3 verdict agreement, original vs independent physics chain (cloud + temperature claims) ===")
    t = phys.groupby("site").apply(lambda g: pd.Series({"n": len(g), "agree": (g.orig == g.alt).mean(),
                                                         "pass_orig": (g.orig == "PASS").mean(), "pass_alt": (g.alt == "PASS").mean(),
                                                         "kappa": kappa(g.orig, g.alt),
                                                         "true_effect_corr": np.corrcoef(g.true_orig, g.true_alt)[0, 1]}))
    print(t.round(3).to_string())
    print(pd.crosstab(phys.orig, phys.alt, margins=True).to_string())
    return a


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    po = (a.to_numpy() == b.to_numpy()).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def measured_gamma():
    """Temperature coefficient the physics chain assumes (-0.37 %/K) vs measured: clear high-irradiance hours, power per
    unit plane-of-array irradiance against cell temperature (DKASC: SAPM from measured POA/air/wind; Henderson: module sensor)."""
    import pvlib
    out = {}
    c = pd.read_parquet(BD.CLEAN)
    c = c[c.valid & c.poa_meas.notna() & (c.poa_meas > 800) & c.power.notna()]
    tc = pvlib.temperature.sapm_cell(c.poa_meas, c.temp, c.wind.fillna(1.0), -3.56, -0.075, 3)
    out["dkasc38 (SAPM cell temp)"] = fit_gamma(c.power / c.poa_meas, tc, c.index.normalize())
    h = pd.read_csv("data/interim/pvdaq_1423_hourly.csv", index_col=0, parse_dates=True)
    h = h[(h.n_min >= 50) & (h.poa > 800) & h.t_mod.notna() & (h.p_w > 0)]
    out["henderson (module sensor)"] = fit_gamma(h.p_w / h.poa, h.t_mod, h.index.normalize())
    g = pd.DataFrame(out, index=["gamma_per_K", "ci_lo", "ci_hi", "days"]).T
    g.to_csv(OUT / "measured_gamma.csv")
    print("\nassumed gamma -0.0037 /K; measured:\n", g.to_string())
    return g


def fit_gamma(ratio, tc, days, n_boot=500):
    """ratio = a * (1 + gamma * (Tc - 25)) -> gamma = slope / intercept-at-25; CI by resampling whole days
    (samples within a day are not independent)."""
    x, y, g = np.asarray(tc, float), np.asarray(ratio, float), np.asarray(days)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    q = np.quantile(y, [0.05, 0.95])  # drop partial-shade / sensor-glitch tails
    keep = (y > q[0]) & (y < q[1])
    x, y, g = x[keep], y[keep], g[keep]
    est = lambda xx, yy: (lambda p: p[0] / (p[1] + 25 * p[0]))(np.polyfit(xx, yy, 1))
    uniq, inv = np.unique(g, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    bs = [(lambda i: est(x[i], y[i]))(np.concatenate([idx[k] for k in rng.integers(0, len(uniq), len(uniq))])) for _ in range(n_boot)]
    return est(x, y), *np.percentile(bs, [2.5, 97.5]), len(uniq)


if __name__ == "__main__":
    from src.experiments.gate_eval import setup
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    df, m, F = setup()
    b = bases(df, F) if step in ("bases", "all") or not (OUT / "bases.csv").exists() else pd.read_csv(OUT / "bases.csv", parse_dates=["day"])
    if step in ("planted", "all"):
        planted(b)
    if step in ("sweep", "all"):
        sweep(b)
    if step in ("reference", "all"):
        measured_gamma()
        reference(df, F)
    if step in ("figure", "all"):
        figure()


def figure():
    """Phase 1 summary: planted errors, the gate's resolution, robustness of the real-claim conclusion, independent physics."""
    import matplotlib.pyplot as plt
    from src.experiments.plot_gate import INK, MUTED, S1, S2, S3, _style
    s = pd.read_csv(OUT / "planted_claims.csv")
    curve = pd.read_csv(OUT / "planted_size_curve.csv")
    real = pd.read_csv(OUT / "sweep_real.csv")
    alt = pd.read_csv(OUT / "reference_alt_chain.csv")
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8), gridspec_kw={"width_ratios": [1.3, 1, 1, 0.9]})

    ax = axs[0]
    kinds = ["correct (±10%)", "sign flipped", "size ×3", "size ×2", "size ×1.5", "size ×0.5", "wrong cause", "wrong window"]
    caught = (s.verdict != "PASS").groupby([s.kind, s.site]).mean().unstack()
    tot = (s.verdict != "PASS").groupby(s.kind).mean()
    y = np.arange(len(kinds))[::-1]
    ax.barh(y, 100 * tot.reindex(kinds), color=[S3] + [S1] * (len(kinds) - 1), height=0.6)
    for site, mk in zip(SITES, "o^sD"):
        ax.plot(100 * caught[site].reindex(kinds), y, mk, ms=4, color=INK, alpha=0.6, label={"dkasc38": "Alice Springs", "elb": "ELB"}.get(site, site.title()))
    for yi, v in zip(y, tot.reindex(kinds)):
        ax.text(103, yi, f"{100 * v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, ["correct claim\n(false alarm)"] + kinds[1:], fontsize=8.5)
    ax.set_xlim(0, 112); ax.set_xlabel("% of planted claims the gate does not pass", fontsize=8.5, color=MUTED)
    ax.set_title("a  Planted claims caught (4 sites, 23k claims)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4); _style(ax)

    ax = axs[1]
    c = curve[curve.ratio.between(-1, 3)]
    ax.axvspan(0.75, 1.25, color=S3, alpha=0.12)
    ax.text(1.0, 40, "tolerance\n±25% or\n0.2 kWh", ha="center", fontsize=7.5, color=MUTED)
    ax.plot(c.ratio, 100 * c.caught, color=S1, lw=2)
    ax.set_xlabel("claimed effect ÷ true effect  (−1 = sign flipped)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("% caught", fontsize=8.5, color=MUTED)
    ax.set_title("b  Resolution: what size of error is caught", fontsize=10, color=INK, loc="left"); _style(ax)

    ax = axs[2]
    names = {"dkasc38": "Alice Springs", "henderson": "Henderson", "golden": "Golden", "elb": "ELB"}
    d = real[real.material == 0.1]
    x = np.arange(len(SITES))
    for j, (col, lab, colr) in enumerate([("B2 + physics explainer", "physics explainer", S1), ("B2 + SHAP", "SHAP", S2)]):
        g = d.groupby("site")[col]
        dflt = d[(d.tol_abs == 0.2) & (d.tol_rel == 0.25)].set_index("site")[col]
        xs = x + (j - 0.5) * 0.3
        ax.bar(xs, 100 * dflt.reindex(SITES), width=0.28, color=colr, label=lab)
        ax.errorbar(xs, 100 * dflt.reindex(SITES), yerr=[100 * (dflt - g.min()).reindex(SITES), 100 * (g.max() - dflt).reindex(SITES)],
                    fmt="none", color=INK, lw=1, capsize=3)
    ax.set_xticks(x, [names[s_] for s_ in SITES], fontsize=8.5)
    ax.set_ylabel("% of B2 claims passing", fontsize=8.5, color=MUTED)
    ax.set_title("c  Default settings (bars) and range over 25 tolerances", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False); _style(ax)

    ax = axs[3]
    p = alt[alt.driver != "ml_correction"]
    agree = p.groupby("site").apply(lambda g: (g.orig == g.alt).mean()).reindex(SITES)
    ax.bar(x, 100 * agree, color=S3, width=0.55)
    for xi, v in zip(x, agree):
        ax.text(xi, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [names[s_] for s_ in SITES], fontsize=8.5); ax.set_ylim(0, 110)
    ax.set_ylabel("% same verdict", fontsize=8.5, color=MUTED)
    ax.set_title("d  Same verdict with independent physics", fontsize=10, color=INK, loc="left"); _style(ax)
    fig.suptitle("Gate robustness: planted errors, resolution, threshold sensitivity, independent physics reference", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(OUT / "fig_gate_robustness.png", dpi=180); plt.close(fig)
    print("saved", OUT / "fig_gate_robustness.png")
