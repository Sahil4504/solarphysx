"""Re-score LLM first drafts (round 0) with the current gate, per site, and split errors by where the number came from.

Round 0 = the LLM writing without verification. Re-scoring needs no LLM: the logs keep every drafted claim.
Number source compares each draft with the evidence it was given (= template writer's round 0, which is the evidence).
  python -m src.agent.rescore            -> results/agent/qwen_rescored_summary.csv (+ printed table)
"""

import numpy as np
import pandas as pd

import src.data.build_dayahead as BD
from src.agent.explainer import FORECASTER, OUT, critique
from src.gate import gate as G

RUNS = {"dkasc38": "hf_Qwen2.5-3B-Instruct", "henderson": "hf_Qwen2.5-3B-Instruct_henderson",
        "golden": "hf_Qwen2.5-3B-Instruct_golden", "elb": "hf_Qwen2.5-3B-Instruct_elb"}
TEMPLATE = {"dkasc38": "template", "henderson": "template_henderson", "golden": "template_golden", "elb": "template_elb"}


def source(r0: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    r = r0.merge(ev, on=["day", "window", "driver"], how="left")
    return pd.Series(np.where(r.ev.isna(), "not in evidence", np.where((r.delta_kwh - r.ev).abs() <= 0.011, "copied evidence",
                     np.where((r.delta_kwh + r.ev).abs() <= 0.011, "sign flipped", "number changed"))), index=r0.index)


def day_ci(flags: pd.Series, days: pd.Series, n=1000, seed=0):
    g = flags.groupby(days.to_numpy()).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    bs = [(lambda s: s["sum"].sum() / s["size"].sum())(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    from src.experiments.gate_eval import setup
    df, m, F = setup()
    f, ref = F[FORECASTER], G.references(df[df.split == "train"])
    rows_out = []
    for site, run in RUNS.items():
        if site == "dkasc38":
            BD.use_site("dkasc38"); te = df[df.split == "test"]
        else:
            BD.use_site(site); te = pd.read_parquet(BD.SITES[site]["out"].replace(".parquet", "_cal.parquet"))
        by_day = {f"{k:%Y-%m-%d}": g for k, g in te.groupby("day")}
        L = pd.read_csv(OUT / f"{run}_log.csv")
        r0 = L[(L["round"] == 0) & (L.verdict != "INVALID")].copy()
        recs = []
        for _, c in r0.iterrows():  # same critic the agent uses, now with gate v1.1
            claim = {"window": c.window, "driver": c.driver, "delta_kwh": float(c.delta_kwh), "sentence": str(c.sentence)}
            recs.append(critique([claim], by_day[c.day], f, ref)[0][0]["verdict"])
        r0["v11"] = recs
        ev = pd.read_csv(OUT / f"{TEMPLATE[site]}_log.csv").query("round == 0")[["day", "window", "driver", "delta_kwh"]].rename(columns={"delta_kwh": "ev"})
        r0["source"] = source(r0, ev)
        n_inv = int(((L["round"] == 0) & (L.verdict == "INVALID")).sum())
        wrong = r0.v11 != "PASS"
        lo, hi = day_ci(wrong, r0.day)
        shown = len(pd.read_csv(OUT / f"{run}_shown.csv"))
        rows_out.append(dict(site=site, days=r0.day.nunique(), drafted=len(r0), invalid=n_inv,
                             v1_wrong=round((r0.verdict != "PASS").mean(), 3), v11_wrong=round(wrong.mean(), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             v11_flag=round((r0.v11 == "FLAG").mean(), 3), v11_block=round((r0.v11 == "BLOCK").mean(), 3),
                             copied=round((r0.source == "copied evidence").mean(), 3),
                             wrong_from_evidence=round((wrong & (r0.source == "copied evidence")).mean(), 3),
                             wrong_from_llm=round((wrong & (r0.source != "copied evidence")).mean(), 3),
                             llm_changed_pass=round((r0[r0.source != "copied evidence"].v11 == "PASS").mean(), 3), shown=shown))
        print(site, pd.crosstab(r0.source, r0.v11).to_string(), "\n")
    out = pd.DataFrame(rows_out).set_index("site")
    out.to_csv(OUT / "qwen_rescored_summary.csv")
    print(out.T.to_string())
