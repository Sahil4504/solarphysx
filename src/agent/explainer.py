"""Agentic explainer, phase 1: explain -> verify -> revise, with the verification gate as critic.

Per forecast day the writer (an LLM, or the deterministic 'template' stand-in) gets structured evidence
(window energies + physics-chain effects), returns JSON claims with one sentence each, and the critic runs
gate T2 (physical sign) + T3 (counterfactual re-forecast of the real model) + a text check (sentence states
the claimed number and direction). Failed claims go back as feedback with the gate's reason and true value.
After max_rounds anything not PASS is withheld (abstain): the operator only sees verified sentences.

  python -m src.agent.explainer --backend template --every 10           (CPU, no LLM: tests the loop)
  python -m src.agent.explainer --backend hf --every 10                 (local LLM on GPU; needs `pip install transformers accelerate`)
  python -m src.agent.explainer --selfcheck
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.gate import gate as G

OUT = Path("results/agent")
FORECASTER = "B2_physics_constrained"
REFERENCE = {"cloud": "clear_sky", "temperature": "t25", "ml_correction": "none"}  # driver -> gate reference scenario
LOWER = re.compile(r"\b(lower|lowers|lowered|reduce|reduces|reduced|reduction|cut|cuts|decrease|decreases|drop|drops|less|loses|loss|costs)\b", re.I)
CAUSE = {"cloud": r"cloud", "temperature": r"temperat|°c|\bheat|\bhot|\bwarm|\bcool", "ml_correction": r"correction|learned|\bml\b"}
RAISE = re.compile(r"\b(raise|raises|raised|increase|increases|increased|add|adds|boost|boosts|more|gains|extra)\b", re.I)

SYSTEM = """You explain a day-ahead solar PV forecast (DKASC Alice Springs, 5.9 kW array) to a grid operator.
You get evidence for three windows (morning 06-10, midday 10-14, afternoon 14-19 ACST). Make at most one claim per
(window, driver) for drivers: cloud (effect vs a clear sky), temperature (effect vs 25 C air), ml_correction
(learned correction on top of physics). Skip effects smaller than 0.10 kWh.
Answer with JSON only, no prose outside it:
{"claims": [{"window": "...", "driver": "...", "delta_kwh": <signed number, + raises output>, "sentence": "..."}]}
Each sentence is a full sentence for the operator: it names the window and the cause, states the number
in kWh and says whether it raises or lowers output, e.g. "In the midday, cloud lowers output by 2.10 kWh compared with a clear sky."
A verifier re-runs the forecast model to check every claim; if it sends corrections, fix those claims and resend
the full JSON. Drop a claim if you cannot fix it."""


# ---------------- evidence ----------------
def evidence(rows, forecaster, derate) -> dict:
    """What the writer may use: per window energies, weather, and the physics chain's effect estimates."""
    draft = {(c.window, c.driver): c.delta_kwh for c in G.explain_physics(rows, forecaster, derate, "agent")}
    ev = {"day": f"{rows.day.iloc[0]:%Y-%m-%d}", "windows": {}}
    for w in G.WINDOWS:
        r = G.window_rows(rows, w)
        ev["windows"][w] = {
            "forecast_kwh": round(float(np.clip(forecaster(r), 0, None).sum()), 2),
            "clear_sky_kwh": round(float(derate * r.p_cs1.sum()), 2),
            "gfs_clearness": round(float(r.k.mean()), 2), "air_temp_c": round(float(r.t2m.mean()), 1),
            "physics_estimates_kwh": {d: round(v, 2) for (ww, d), v in draft.items() if ww == w}}
    return ev


# ---------------- critic ----------------
def parse(text: str):
    """JSON claims from the writer's reply; raises ValueError with a reason the writer can act on."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(re.sub(r":\s*\+(?=\d)", ": ", m.group(0)))  # LLMs write "+0.87", which JSON forbids
    claims = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list")
    for c in claims:  # anything malformed becomes feedback to the writer, never a crash
        if not isinstance(c, dict) or c.get("window") not in G.WINDOWS or c.get("driver") not in REFERENCE:
            raise ValueError(f"bad window/driver in {c}")
        try:
            c["delta_kwh"] = float(c.get("delta_kwh"))
        except (TypeError, ValueError):
            raise ValueError(f"delta_kwh must be a number in {c}") from None
        if not np.isfinite(c["delta_kwh"]):
            raise ValueError(f"delta_kwh must be finite in {c}")
        c["sentence"] = str(c.get("sentence", ""))
    return claims


def sentence(window, driver, delta):
    """Canonical operator sentence; the critic sends it back as the fix for text errors."""
    verb = "raises" if delta > 0 else "lowers"
    ref = {"cloud": "a clear sky", "temperature": "25 °C air", "ml_correction": "pure physics"}[driver]
    return f"In the {window}, {driver.replace('_', ' ')} {verb} output by {abs(delta):.2f} kWh compared with {ref}."


def text_check(sentence: str, delta: float, driver=None, window=None):
    """Material claim; sentence states |delta| (1-decimal rounding), a direction that matches, the cause and the window."""
    if abs(delta) < G.MATERIAL_KWH:
        return f"effect below {G.MATERIAL_KWH:.2f} kWh: drop this claim"
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", sentence)]
    if not any(abs(x - abs(delta)) <= 0.05 for x in nums):
        return f"sentence does not state {abs(delta):.2f} kWh"
    lo, hi = bool(LOWER.search(sentence)), bool(RAISE.search(sentence))
    if not (lo or hi):
        return "sentence does not say whether output is raised or lowered"
    if lo != hi and (hi if delta < 0 else lo):
        return "sentence direction contradicts the sign of delta_kwh"
    if driver and not re.search(CAUSE[driver], sentence, re.I):
        return f"sentence does not name the cause ({driver})"
    if window and window not in sentence.lower():
        return f"sentence does not name the window ({window})"
    return None


def critique(claims, rows, forecaster, ref, model="B2"):
    """Gate verdict per claim; returns records and the feedback items for non-PASS claims."""
    recs, fb = [], []
    for i, c in enumerate(claims):
        cl = G.Claim(rows.day.iloc[0], c["window"], c["driver"], REFERENCE[c["driver"]], c["delta_kwh"], "agent", model)
        v2 = G.t2_plausibility(cl, rows, ref)
        v3, true = G.t3_truth(cl, rows, forecaster, ref)
        why = text_check(c["sentence"], c["delta_kwh"], c["driver"], c["window"])
        verdict = max(v2, v3, "FLAG" if why else "PASS", key=G.RANK.get)
        fix = true if v3 != "PASS" else c["delta_kwh"]  # value the sentence should carry
        problems = [v2 == "BLOCK" and "physically wrong direction",
                    v2 == "FLAG" and "learned correction too large relative to physics",
                    v3 != "PASS" and f"re-forecast with the model says {true:+.2f} kWh, you claimed {c['delta_kwh']:+.2f}", why]
        action = ("drop this claim" if v2 != "PASS" or abs(fix) < G.MATERIAL_KWH
                  else f'write exactly: "{sentence(c["window"], c["driver"], fix)}" with delta_kwh {fix:+.2f}')
        reason = "; ".join(p for p in problems if p) + "; " + action
        recs.append({**c, "true_kwh": round(true, 3), "t2": v2, "t3": v3, "text": why or "ok", "verdict": verdict})
        if verdict != "PASS":
            fb.append({"claim": i, "window": c["window"], "driver": c["driver"], "verdict": verdict,
                       "true_kwh": round(true, 2), "reason": reason})
    return recs, fb


def feedback_message(fb) -> str:
    return ("The verifier rejected some claims. Fix them (use the re-forecast value) or drop them, keep the PASS claims "
            "unchanged, and resend the full JSON.\n" + json.dumps({"rejected": fb}))


# ---------------- writers ----------------
def template_writer(messages) -> str:
    """Deterministic stand-in for the LLM: physics estimates first, then adopts the critic's true value (drops BLOCKs)."""
    ev = json.loads(re.search(r"\{.*\}", messages[1]["content"], re.S).group(0))
    if len(messages) == 2:
        claims = [{"window": w, "driver": d, "delta_kwh": v} for w, x in ev["windows"].items()
                  for d, v in x["physics_estimates_kwh"].items()]
    else:
        claims = json.loads(messages[-2]["content"])["claims"]
        fb = json.loads(re.search(r"\{.*\}", messages[-1]["content"], re.S).group(0)).get("rejected", [])
        drop = set()
        for f in fb:
            if f["verdict"] == "BLOCK" or abs(f["true_kwh"]) < 0.10 or "too large" in f["reason"]:
                drop.add(f["claim"])
            else:
                claims[f["claim"]]["delta_kwh"] = f["true_kwh"]
        claims = [c for i, c in enumerate(claims) if i not in drop]
    for c in claims:
        c["sentence"] = sentence(c["window"], c["driver"], c["delta_kwh"])
    return json.dumps({"claims": claims})


def hf_writer(model_id="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=700):
    """Local chat LLM via transformers, greedy decoding (deterministic, reproducible)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_id)
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    def write(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
        out = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return write


# ---------------- loop ----------------
def run_day(rows, forecaster, derate, ref, writer, max_rounds=3):
    """explain -> verify -> revise. Returns one record per claim per round, and the final shown claims."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Evidence for tomorrow's forecast:\n" + json.dumps(evidence(rows, forecaster, derate))}]
    log, recs = [], []
    for rnd in range(max_rounds + 1):
        reply = writer(msgs)
        try:
            claims = parse(reply)
        except (ValueError, json.JSONDecodeError) as e:  # malformed output is feedback too, never shown
            log.append({"round": rnd, "verdict": "INVALID", "reason": str(e)[:200]})  # keep the last valid round's verified claims
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"Invalid output ({e}). Resend JSON only."}]
            continue
        recs, fb = critique(claims, rows, forecaster, ref)
        log += [{"round": rnd, **r} for r in recs]
        if not fb:
            break
        msgs += [{"role": "assistant", "content": json.dumps({"claims": claims})}, {"role": "user", "content": feedback_message(fb)}]
    shown = [r for r in recs if r["verdict"] == "PASS"]
    return log, shown


def run(writer, every=10, max_rounds=3, name="template"):
    from src.experiments.gate_eval import setup
    df, m, F = setup()
    f, d = F[FORECASTER], m["derate"]
    ref = G.references(df[df.split == "train"])
    te = df[df.split == "test"]
    days = [g for _, g in te.groupby("day")][::every]
    OUT.mkdir(parents=True, exist_ok=True)
    lp, sp = OUT / f"{name}_log.csv", OUT / f"{name}_shown.csv"
    # resumes an interrupted run: days already in the log are skipped (rename old files after changing the code)
    read = lambda p: pd.read_csv(p).to_dict("records") if p.exists() and p.stat().st_size > 2 else []
    logs, shown = read(lp), read(sp)
    done = {r["day"] for r in logs}
    for i, rows in enumerate(days):
        day = f"{rows.day.iloc[0]:%Y-%m-%d}"
        if day in done:
            continue
        log, sh = run_day(rows, f, d, ref, writer, max_rounds)
        logs += [{"day": day, **r} for r in log]
        shown += [{"day": day, **r} for r in sh]
        pd.DataFrame(logs).to_csv(lp, index=False); pd.DataFrame(shown).to_csv(sp, index=False)  # saved every day
        print(f"{i + 1}/{len(days)} {day}: rounds {max(r['round'] for r in log) + 1}, shown {len(sh)}", flush=True)
    log, sh = pd.DataFrame(logs), pd.DataFrame(shown)
    summary(log, name)
    return log, sh


def summary(log: pd.DataFrame, name: str):
    """Round-0 verdicts (what an unverified explainer would show) vs final round, and abstentions."""
    last = log.groupby("day")["round"].transform("max")
    r0, fin = log[log["round"] == 0], log[log["round"] == last]
    s = pd.DataFrame({"round 0 (unverified draft)": r0.verdict.value_counts(normalize=True),
                      "final round": fin.verdict.value_counts(normalize=True)}).fillna(0).round(3)
    rounds = log.groupby("day")["round"].max() + 1
    txt = (f"[{name}] days {log.day.nunique()}, claims round 0: {len(r0)}, final: {len(fin)}, "
           f"shown (PASS): {(fin.verdict == 'PASS').sum()}, rounds used mean {rounds.mean():.2f} max {rounds.max()}\n{s}")
    print(txt)
    (OUT / f"{name}_summary.txt").write_text(txt, encoding="utf-8")


def selfcheck():
    from src.experiments.gate_eval import setup
    df, m, F = setup()
    f, d = F[FORECASTER], m["derate"]
    ref = G.references(df[df.split == "train"])
    te = df[df.split == "test"]
    rows = min((g for _, g in te.groupby("day")), key=lambda g: g.k.mean())  # cloudiest day: large, signed effects
    # critic catches a wrong sign, a wrong number in the sentence, and accepts a true claim
    w = "midday"
    true = float((f(G.window_rows(rows, w)) - f(G.counterfactual(G.window_rows(rows, w), "cloud", "clear_sky", ref))).sum())
    ok = {"window": w, "driver": "cloud", "delta_kwh": true, "sentence": sentence(w, "cloud", true)}
    bad_sign = {**ok, "delta_kwh": -true}
    bad_text = {**ok, "sentence": sentence(w, "cloud", true + np.sign(true))}
    recs, _ = critique([ok, bad_sign, bad_text], rows, f, ref)
    assert [r["verdict"] for r in recs] == ["PASS", "BLOCK", "FLAG"], recs
    assert parse('{"claims": [{"window": "midday", "driver": "cloud", "delta_kwh": +0.87, "sentence": "x"}]}')[0]["delta_kwh"] == 0.87
    for bad in ['{"claims": [{"window": "midday", "driver": "cloud", "delta_kwh": null}]}', '{"claims": ["x"]}', '{"claims": 3}']:
        try:
            parse(bad); raise AssertionError(bad)
        except ValueError:
            pass
    assert text_check("raises output by 0.40 kWh", -0.4) and text_check("lowers output by 0.4 kWh", -0.4) is None
    assert text_check("-0.37 kWh (lower)", -0.37, "cloud", "morning")  # fragment without cause/window is rejected
    assert text_check(sentence("midday", "cloud", 0.07), 0.07)  # immaterial
    assert text_check("+0.28 kWh", 0.28)  # no direction
    assert text_check("A reduction of 1.14 kWh in the morning due to cloud cover.", -1.14, "cloud", "morning") is None
    with_bad_json = lambda msgs: "sorry, no json" if len(msgs) == 2 else template_writer(msgs[:2])
    log, shown = run_day(rows, f, d, ref, with_bad_json)
    assert log[0]["verdict"] == "INVALID" and all(r["verdict"] == "PASS" for r in shown)
    log, shown = run_day(rows, f, d, ref, template_writer)
    assert shown and all(r["verdict"] == "PASS" for r in shown)
    print(f"self-check ok: critic PASS/BLOCK/FLAG as expected, invalid JSON recovered, {len(shown)} verified claims on {rows.day.iloc[0]:%Y-%m-%d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["template", "hf"], default="template")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--every", type=int, default=10, help="use every n-th test day (1 = all 273)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.backend == "template":
        run(template_writer, a.every, a.rounds, "template")
    else:
        run(hf_writer(a.model), a.every, a.rounds, "hf_" + a.model.split("/")[-1])
