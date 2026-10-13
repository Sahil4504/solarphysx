"""Which monotone constraints on B's residual remove gate T2 violations, and what do they cost? -> results/gate/mono_ablation.csv
Every row: same B (physics + LightGBM residual, same data/split/seed) + night mask; only the constraint set changes.
  python -m src.experiments.mono_ablation
"""

import numpy as np
import pandas as pd

from src.experiments.dayahead_bench import FEATS, _lgb, fit_derate, load
from src.gate import gate as G

ROOT = {"k": 1, "tcc": -1, "t2m": -1}
IRR = {"ghi_gfs": 1, "poa_gfs": 1, "p_phys1": 1}
SETS = {"none (= B + mask)": {}, "t2m": {"t2m": -1}, "k": {"k": 1}, "tcc": {"tcc": -1}, "k+tcc": {"k": 1, "tcc": -1},
        "irradiance trio": IRR, "root causes (k, tcc, t2m)": ROOT, "all six": {**ROOT, **IRR}}

if __name__ == "__main__":
    df = load()
    d = df.dropna(subset=["y"])
    tr, va = d[d.split == "train"], d[d.split == "val"]
    der = fit_derate(df)
    r = d.y - der * d.p_phys1
    te = df[df.split == "test"]
    sc = te[te.daytime & te.y.notna() & te.p_lag.notna()]  # scored like the benchmark (persistence must exist)
    pers = np.sqrt(((sc.p_lag - sc.y) ** 2).mean())
    rows = []
    for name, mono in SETS.items():
        m = _lgb(tr[FEATS], r.loc[tr.index], va[FEATS], r.loc[va.index], objective="regression",
                 monotone_constraints=[mono.get(f, 0) for f in FEATS])
        f = lambda x, m=m: (der * x.p_phys1.to_numpy() + m.predict(x[FEATS])) * (x.cs_ghi.to_numpy() > 0)
        rmse = np.sqrt(((np.clip(f(sc), 0, 5.9) - sc.y) ** 2).mean())
        t2 = G.t2_monotonicity(f, te[te.daytime]).mean() * 100
        rows.append(dict(constraints=name, rmse_kW=rmse, skill_vs_persistence=1 - rmse / pers,
                         clearer_lowers_pct=t2.clearer_lowers, hotter_raises_pct=t2.hotter_raises))
    out = pd.DataFrame(rows).round(4)
    out.to_csv("results/gate/mono_ablation.csv", index=False)
    print(f"test daytime hours: {len(sc)}"); print(out.to_string(index=False))
