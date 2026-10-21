"""SolarPhysX verification gate: checks a day-ahead forecast and its explanation against physics.

2a  Claim            structured explanation unit (driver, window, direction, size in kWh, reference scenario)
    counterfactual   physics-consistent "what if" inputs: change the root cause, recompute GHI -> POA -> power
    explain_shap     standard SHAP narrative (top-2 driver groups per window, relative to an average day)
    explain_physics  B's native decomposition (cloud vs clear sky, temperature vs 25 C, ML correction)
2b  t1_output        forecast within physical bounds (clear-sky envelope, dark hours, ramps, daily energy)
2c  t2_plausibility  claim names a physical cause with the physically right sign; model responds monotonically
2d  t3_truth         claim's size and sign reproduced by re-running the same forecaster on the counterfactual
2e  score_day        Physical Consistency Score (PCS) and PASS / FLAG / BLOCK per forecast day
Self-check:  python -m src.gate.gate
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.build_dayahead as BD  # BD.TZ / BD.pv_unit follow BD.use_site(): counterfactual physics uses the active site
from src.experiments.dayahead_bench import FEATS

WINDOWS = {"morning": (6, 10), "midday": (10, 14), "afternoon": (14, 19)}  # [start, end) local hours
GROUPS = {"cloud": ["k", "tcc"], "irradiance": ["ghi_gfs", "poa_gfs", "p_phys1"], "temperature": ["t2m"],
          "yesterday": ["p_lag", "pcs_lag", "kd_lag"], "sun_geometry": ["p_cs1", "zenith", "hour"]}
PHYSICAL = {"cloud", "irradiance", "temperature", "sun_geometry"}  # 'yesterday', 'ml_correction': correlations, not causes
SIGN_FEATURE = {"irradiance": "p_phys1", "sun_geometry": "p_cs1"}  # more of it than the reference -> more power (cloud/temperature: physics chain)
MATERIAL_KWH = 0.10  # claims smaller than this are not made
TOL_ABS, TOL_REL = 0.20, 0.25  # PASS if |claimed - true| <= max(TOL_ABS, TOL_REL * |true|)
ML_SHARE = 0.25  # ML correction above this share of the window's physics energy is flagged


# ---------------- 2a: claims + physics-consistent counterfactuals ----------------
@dataclass
class Claim:
    day: pd.Timestamp
    window: str
    driver: str  # cloud | temperature | yesterday | sun_geometry | ml_correction
    reference: str  # clear_sky | t25 | average_day | none
    delta_kwh: float  # claimed effect on window energy vs the reference scenario (+ raises, - lowers)
    explainer: str
    model: str

    def text(self):
        ref = {"clear_sky": "a clear sky", "t25": "25 °C air", "average_day": "an average day", "none": "pure physics"}
        verb = "raise" if self.delta_kwh > 0 else "lower"
        return (f"{self.day:%d %b}, {self.window}: {self.driver.replace('_', ' ')} {verb}s output by "
                f"{abs(self.delta_kwh):.2f} kWh compared with {ref[self.reference]}")


def references(train: pd.DataFrame) -> pd.DataFrame:
    """'Average day' = training mean of every feature at the same local hour (same reference as per-hour SHAP)."""
    return train.groupby("hour")[[c for cols in GROUPS.values() for c in cols if c != "hour"]].mean()


def _at_hours(ref: pd.DataFrame, x: pd.DataFrame, col: str) -> np.ndarray:
    return ref[col].reindex(x.hour).to_numpy()


def _physics(x: pd.DataFrame) -> pd.DataFrame:
    """Recompute irradiance and physics power after k or t2m changed (same chain as build_dayahead)."""
    mid = (x.index + pd.Timedelta(minutes=30)).tz_localize(BD.TZ)
    x["ghi_gfs"] = x.k * x.cs_ghi
    poa, p, _ = BD.pv_unit(mid, x.ghi_gfs, x.t2m)
    poa_cs, p_cs, _ = BD.pv_unit(mid, x.cs_ghi, x.t2m)
    x["poa_gfs"], x["p_phys1"], x["poa_cs"], x["p_cs1"] = poa.to_numpy(), p.to_numpy(), poa_cs.to_numpy(), p_cs.to_numpy()
    return x


def counterfactual(x: pd.DataFrame, driver: str, reference: str, ref: dict) -> pd.DataFrame:
    """Inputs for the same hours with one root cause set to its reference; dependent physics recomputed."""
    x = x.copy()
    if driver == "cloud" and reference == "clear_sky":
        x["k"], x["tcc"] = 1.0, 0.0
        return _physics(x)
    if driver == "temperature" and reference == "t25":
        x["t2m"] = 25.0
        return _physics(x)
    assert reference == "average_day", (driver, reference)
    for c in GROUPS[driver]:
        if c != "hour":
            x[c] = _at_hours(ref, x, c)
    # cloud / temperature are root causes: recompute irradiance and physics power from them
    # shortcut: sun_geometry and irradiance are swapped as features (like SHAP does), not re-derived physically
    return _physics(x) if driver in ("cloud", "temperature") else x


def window_rows(day_rows: pd.DataFrame, w: str) -> pd.DataFrame:
    a, b = WINDOWS[w]
    return day_rows[(day_rows.hour >= a) & (day_rows.hour < b)]


def explain_shap(day_rows, phi: pd.DataFrame, model: str):
    """Standard SHAP narrative: per window, the two driver groups with the largest summed attribution."""
    out = []
    for w in WINDOWS:
        g = {d: phi.loc[window_rows(day_rows, w).index, cols].to_numpy().sum() for d, cols in GROUPS.items()}
        for d in sorted(g, key=lambda d: -abs(g[d]))[:2]:
            if abs(g[d]) >= MATERIAL_KWH:
                out.append(Claim(day_rows.day.iloc[0], w, d, "average_day", float(g[d]), "SHAP", model))
    return out


def explain_physics(day_rows, forecaster, derate: float, model: str):
    """Physics-decomposition explainer: what the physics chain says each cause does, plus the ML correction."""
    out = []
    t25 = _physics(day_rows.assign(t2m=25.0))
    for w in WINDOWS:
        r = window_rows(day_rows, w)
        claims = {("cloud", "clear_sky"): derate * (r.p_phys1 - r.p_cs1).sum(),
                  ("temperature", "t25"): derate * (r.p_phys1 - t25.loc[r.index, "p_phys1"]).sum(),
                  ("ml_correction", "none"): (forecaster(r) - derate * r.p_phys1).sum()}
        for (d, ref), v in claims.items():
            if abs(v) >= MATERIAL_KWH:
                out.append(Claim(day_rows.day.iloc[0], w, d, ref, float(v), "physics", model))
    return out


# ---------------- 2b: T1 output plausibility ----------------
@dataclass
class T1Bounds:
    envelope_kw: float  # forecast may exceed clear-sky power by at most this (cloud enhancement seen in training)
    dark_kw: float  # max power in hours with no sun at all
    ramp_kw: float  # max hour-to-hour change beyond the clear-sky change
    day_ratio: float  # max daily energy / clear-sky daily energy

    @classmethod
    def calibrate(cls, train: pd.DataFrame, margin=1.1):
        t = train.dropna(subset=["y"])
        ramp = (t.y.groupby(t.day).diff().abs() - t.p_cs1.groupby(t.day).diff().abs()).max()
        day = t.y.groupby(t.day).sum() / t.p_cs1.groupby(t.day).sum()
        return cls(envelope_kw=margin * (t.y - t.p_cs1).max(), dark_kw=max(0.02, margin * t.y[t.cs_ghi == 0].max()),
                   ramp_kw=margin * ramp, day_ratio=margin * day.max())


def t1_output(pred: pd.Series, rows: pd.DataFrame, b: T1Bounds) -> pd.DataFrame:
    """Per day: which physical bounds the forecast breaks (True = violation)."""
    p = pred.reindex(rows.index)
    over = p - rows.p_cs1 > b.envelope_kw
    dark = (rows.cs_ghi == 0) & (p > b.dark_kw)
    ramp = p.groupby(rows.day).diff().abs() - rows.p_cs1.groupby(rows.day).diff().abs() > b.ramp_kw
    v = pd.DataFrame({"envelope": over, "dark": dark, "ramp": ramp, "day": rows.day}).groupby("day").any()
    v["daily_energy"] = (p.groupby(rows.day).sum() / rows.p_cs1.groupby(rows.day).sum()) > b.day_ratio
    v["t1_fail"] = v.any(axis=1)
    return v


# ---------------- 2c: T2 attribution plausibility ----------------
PHYS_SIGN_KWH = 0.05  # physics effects smaller than this (window energy, before derate) have no direction


def sign_prior(c: Claim, rows: pd.DataFrame, ref: dict):
    """Physically expected direction of a claim (+1/-1), or None when physics gives no direction.
    v1.1: cloud and temperature follow the physics chain's own response to the same counterfactual, so geometry is
    respected (low sun behind a tilted panel: a clearer sky can lower output; v1 assumed 'clearer always raises')."""
    r = window_rows(rows, c.window)
    if c.driver in ("cloud", "temperature"):
        diff = float((r.p_phys1 - counterfactual(r, c.driver, c.reference, ref).p_phys1).sum())
        return None if abs(diff) < PHYS_SIGN_KWH else np.sign(diff)
    if c.driver in SIGN_FEATURE:  # clearer / brighter / higher sun than the reference -> more power
        f = SIGN_FEATURE[c.driver]
        x_ref = 1.0 if c.reference == "clear_sky" else _at_hours(ref, r, f)
        diff = (r[f] - x_ref).mean()
        scale = 0.02 if f == "k" else 0.05  # ignore differences too small to have a direction
        return None if abs(diff) < scale else np.sign(diff)
    return None


_NO_SIGN = object()


def t2_plausibility(c: Claim, rows, ref, s=_NO_SIGN) -> str:
    """BLOCK: physically wrong direction. FLAG: driver is not a physical cause. PASS otherwise.
    s: precomputed sign_prior (robustness experiments re-score many claims for the same window and driver)."""
    s = sign_prior(c, rows, ref) if s is _NO_SIGN else s
    if s is not None and np.sign(c.delta_kwh) != s:
        return "BLOCK"
    if c.driver == "ml_correction":  # a small learned correction is fine; a large one means the model overrides physics
        return "PASS" if abs(c.delta_kwh) <= ML_SHARE * window_rows(rows, c.window).p_phys1.sum() else "FLAG"
    return "PASS" if c.driver in PHYSICAL else "FLAG"


def t2_monotonicity(forecaster, rows: pd.DataFrame, dk=0.1, dt=5.0, tol=0.02) -> pd.DataFrame:
    """Model-level: the model must not move against the physics chain under the same nudge (clearer sky / hotter air).
    v1.1: a drop is only a violation where physics itself does not drop (low sun behind the panel can make a
    clearer sky lower output); likewise a rise with hotter air only where physics does not rise."""
    base = forecaster(rows)
    xk = _physics(rows.assign(k=(rows.k + dk).clip(upper=1.2)))
    xt = _physics(rows.assign(t2m=rows.t2m + dt))
    up_k, up_t = forecaster(xk), forecaster(xt)
    phys_k, phys_t = xk.p_phys1.to_numpy() - rows.p_phys1.to_numpy(), xt.p_phys1.to_numpy() - rows.p_phys1.to_numpy()
    return pd.DataFrame({"clearer_lowers": (up_k < base - tol) & (phys_k >= 0),
                         "hotter_raises": (up_t > base + tol) & (phys_t <= 0)}, index=rows.index)


# ---------------- 2d: T3 claim truth ----------------
def t3_truth(c: Claim, rows, forecaster, ref) -> tuple[str, float]:
    """Re-run the forecaster with the claimed cause set to the claim's reference; compare the change."""
    r = window_rows(rows, c.window)
    if c.driver == "ml_correction":  # read directly from the model, true by construction
        return "PASS", c.delta_kwh
    true = float((forecaster(r) - forecaster(counterfactual(r, c.driver, c.reference, ref))).sum())
    return t3_verdict(c.delta_kwh, true), true


def t3_verdict(delta: float, true: float, tol_abs: float = TOL_ABS, tol_rel: float = TOL_REL) -> str:
    """BLOCK: claimed direction contradicts a material true effect. FLAG: size outside tolerance. PASS otherwise."""
    if np.sign(true) != np.sign(delta) and abs(true) >= tol_abs:
        return "BLOCK"
    return "PASS" if abs(delta - true) <= max(tol_abs, tol_rel * abs(true)) else "FLAG"


# ---------------- 2e: score + verdict ----------------
SCORE = {"PASS": 1.0, "FLAG": 0.5, "BLOCK": 0.0}
RANK = {"PASS": 0, "FLAG": 1, "BLOCK": 2}


def verify(claims, rows_by_day, forecaster, ref, t1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run T2 + T3 on every claim; combine with T1 into a per-day PCS and verdict."""
    recs = []
    for c in claims:
        rows = rows_by_day[c.day]
        v2 = t2_plausibility(c, rows, ref)
        v3, true = t3_truth(c, rows, forecaster, ref)
        recs.append({**asdict(c), "text": c.text(), "true_kwh": true, "t2": v2, "t3": v3,
                     "verdict": max(v2, v3, key=RANK.get)})
    cl = pd.DataFrame(recs)
    days = t1[["t1_fail"]].copy()
    g = cl.groupby("day")
    days["n_claims"] = g.size()
    days["claim_score"] = g.verdict.apply(lambda s: s.map(SCORE).mean())
    days["worst_claim"] = g.verdict.apply(lambda s: max(s, key=RANK.get))
    days["n_claims"] = days.n_claims.fillna(0).astype(int)
    days["pcs"] = np.where(days.n_claims > 0, (1 - days.t1_fail) * 0.25 + days.claim_score.fillna(1) * 0.75,
                           1 - days.t1_fail)  # T1 weighted 1/4, claims 3/4
    days["verdict"] = np.where(days.t1_fail, "BLOCK", days.worst_claim.fillna("PASS"))
    return cl, days


if __name__ == "__main__":
    # self-check on the pure physics forecaster: physics claims about it must all verify exactly
    from src.experiments.dayahead_bench import fit_derate, load
    df = load()
    d = fit_derate(df)
    phys = lambda x: d * x.p_phys1.to_numpy()
    ref = references(df[df.split == "train"])
    te = df[df.split == "test"]
    days = {k: g for k, g in te.groupby("day")}
    some = list(days)[::30]
    claims = [c for k in some for c in explain_physics(days[k], phys, d, "pvlib_physical")]
    for c in claims:
        v3, true = t3_truth(c, days[c.day], phys, ref)
        assert v3 == "PASS" and (c.driver == "ml_correction" or abs(true - c.delta_kwh) < 1e-6), (c, true)
    # counterfactual identity: reference equal to the actual values leaves the forecast unchanged
    r = days[some[0]]
    same = _physics(r.copy())
    assert np.allclose(phys(same), phys(r), atol=1e-6)
    mono = t2_monotonicity(phys, te[te.daytime])
    assert not mono.any().any(), mono.mean()
    assert all(t2_plausibility(c, days[c.day], ref) != "BLOCK" for c in claims)  # physics never contradicts itself
    # v1.1 geometry case: at a US site physics itself lowers output with a clearer sky at low sun; still no violation
    import src.data.build_dayahead as BDS
    us = pd.read_parquet(BDS.SITES["henderson"]["out"]) if Path(BDS.SITES["henderson"]["out"]).exists() else None
    if us is not None:
        BDS.use_site("henderson")
        us = us[us.zenith < 85]
        assert not t2_monotonicity(phys, us).any().any()
        BDS.use_site("dkasc38")
    b = T1Bounds.calibrate(df[df.split != "test"])
    v = t1_output(te.y, te, b)  # reality on unseen days should rarely break bounds calibrated on training
    print(f"self-check ok: {len(claims)} physics claims verified exactly; bounds {b}; "
          f"test actuals breaking T1 on {v.t1_fail.mean():.1%} of days")
