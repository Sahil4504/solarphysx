"""Evaluate the verification gate on the day-ahead test period (Jan-Sep 2023). Writes results/gate/.

Forecasters: B (chosen), A, pvlib physical; T1 also on every saved forecast.
Explainers: SHAP narrative on A and on B; physics decomposition on B (the SolarPhysX explainer).
  python -m src.experiments.gate_eval            (needs results/dayahead_gpu/pred_*.parquet)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.dayahead_bench import FEATS, lightgbm_models, load
from src.gate import gate as G

PRED, OUT = Path("results/dayahead_gpu"), Path("results/gate")


def setup():
    df = load()
    m = lightgbm_models(df, save_preds=False)
    d = m["derate"]
    F = {"B_physics_residual": lambda x: d * x.p_phys1.to_numpy() + m["B_resid"].predict(x[FEATS]),
         "A_lgbm_quantile": lambda x: np.median([m[q].predict(x[FEATS]) for q in ("A10", "A50", "A90")], axis=0),  # = saved P50 after de-crossing
         "pvlib_physical": lambda x: d * x.p_phys1.to_numpy()}
    te = df[df.split == "test"]
    for k in F:  # refit must reproduce the benchmark forecasts exactly (LightGBM is deterministic)
        saved = pd.read_parquet(PRED / f"pred_{k}.parquet").p50.loc[te.index]
        assert np.allclose(np.clip(F[k](te), 0, 5.9), saved, atol=1e-6), k
    return df, m, F


def stage_t1(df):
    b = G.T1Bounds.calibrate(df[df.split != "test"])
    te = df[df.split == "test"]
    rows = {"measured (reality)": G.t1_output(te.y, te, b).mean()}
    for f in sorted(PRED.glob("pred_*.parquet")):
        rows[f.stem[5:]] = G.t1_output(pd.read_parquet(f).p50, te, b).mean()
    t1 = (100 * pd.DataFrame(rows).T).round(1)
    t1.to_csv(OUT / "t1_violations_pct_days.csv")
    return b, t1


def shap_phi(df, m, n_bg=100):  # shap caps interventional backgrounds at 100
    """Interventional SHAP with a same-hour background, so attributions are 'vs an average day at this hour'."""
    import shap
    te, tr = df[df.split == "test"], df[df.split == "train"]
    phi = {"A_lgbm_quantile": [], "B_physics_residual": []}
    for h, x in te.groupby("hour"):
        bg = tr[tr.hour == h][FEATS].sample(min(n_bg, (tr.hour == h).sum()), random_state=0)
        # shortcut: explains the P50 model; A's forecast is the de-crossed median (identical on 92% of hours, <0.03 kW on 99%)
        for k, model in [("A_lgbm_quantile", m["A50"]), ("B_physics_residual", m["B_resid"])]:
            ex = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
            # shap's own additivity check trips on rows with missing lags (LightGBM routes NaN its own way); checked below
            v = ex.shap_values(x[FEATS], check_additivity=False)
            full = x[FEATS].notna().all(axis=1).to_numpy()
            err = np.abs(ex.expected_value + v.sum(1) - model.predict(x[FEATS]))[full]
            assert err.size == 0 or err.max() < 0.05, (k, h, err.max())  # kW; shap casts to float32 (split ties); claims start at 0.1 kWh
            phi[k].append(pd.DataFrame(v, index=x.index, columns=FEATS))
    phi = {k: pd.concat(v).loc[te.index] for k, v in phi.items()}
    ref_p = tr.groupby("hour").p_phys1.mean().reindex(te.hour).to_numpy()
    phi["B_physics_residual"]["p_phys1"] += m["derate"] * (te.p_phys1.to_numpy() - ref_p)  # B's linear physics term
    return phi


def stage_claims(df, m, F, phi):
    te = df[df.split == "test"]
    by_day = {k: g for k, g in te.groupby("day")}
    sets = {"B + physics explainer": [c for g in by_day.values() for c in G.explain_physics(g, F["B_physics_residual"], m["derate"], "B")],
            "B + SHAP": [c for g in by_day.values() for c in G.explain_shap(g, phi["B_physics_residual"], "B")],
            "A + SHAP": [c for g in by_day.values() for c in G.explain_shap(g, phi["A_lgbm_quantile"], "A")]}
    return by_day, sets


def stage_verify(df, F, by_day, sets, b):
    ref = G.references(df[df.split == "train"])
    te = df[df.split == "test"]
    fkey = {"B": "B_physics_residual", "A": "A_lgbm_quantile"}
    claims, days = [], []
    for name, cl in sets.items():
        f = F[fkey[cl[0].model]]
        t1 = G.t1_output(pd.Series(np.clip(f(te), 0, 5.9), index=te.index), te, b)
        c, d = G.verify(cl, by_day, f, ref, t1)
        claims.append(c.assign(setup=name)); days.append(d.assign(setup=name))
    claims, days = pd.concat(claims, ignore_index=True), pd.concat(days)
    claims.to_csv(OUT / "claims.csv", index=False); days.to_csv(OUT / "days.csv")
    mono = {k: G.t2_monotonicity(F[k], te[te.daytime]).mean() * 100 for k in F}
    pd.DataFrame(mono).T.round(2).to_csv(OUT / "t2_monotonicity_pct_hours.csv")
    return claims, days, pd.DataFrame(mono).T.round(2)


def daily_error(df, model):
    te = df[(df.split == "test") & df.daytime & df.y.notna()]
    p = pd.read_parquet(PRED / f"pred_{model}.parquet").p50.loc[te.index]
    return ((p - te.y) ** 2).groupby(te.day).mean() ** 0.5


def risk_coverage(days, err):
    """Accept days in order of decreasing PCS; RMSE of the accepted set at each coverage."""
    d = days.join(err.rename("rmse_day"), how="inner").sort_values("pcs", ascending=False, kind="stable")
    n = np.arange(1, len(d) + 1)
    return pd.DataFrame({"coverage": n / len(d), "rmse": np.sqrt(np.cumsum(d.rmse_day.to_numpy() ** 2) / n)})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df, m, F = setup()
    b, t1 = stage_t1(df)
    print(t1.to_string())
    phi = shap_phi(df, m)
    by_day, sets = stage_claims(df, m, F, phi)
    claims, days, mono = stage_verify(df, F, by_day, sets, b)
    print(mono.to_string())
    print(pd.crosstab(claims.setup, claims.verdict, normalize="index").round(3))
