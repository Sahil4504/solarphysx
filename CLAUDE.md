# CLAUDE.md — SolarPhysX (read fully before doing anything)

> Auto-loaded by Claude Code at session start. This is the complete project handoff as of
> **2026-09-22 (Tuesday)**. The Obsidian vault at `E:\XAI research\SolarPhysX_Vault` is the
> living record; this file is the self-contained snapshot. If they disagree, the vault's
> `00_Project_State/Changelog.md` (newest entries) wins, and this file should be updated.

---

## 0. Start-of-session checklist

**Sessions run in the Claude desktop app (Code tab)** with the working folder set to the parent
`E:\XAI research` (so the vault is reachable; `/add-dir` isn't available in the desktop app).
A short pointer `E:\XAI research\CLAUDE.md` sends you here.

1. Read this file fully.
2. Read `SolarPhysX_Vault/00_Project_State/Project State Summary.md` and the top 3 entries of
   `00_Project_State/Changelog.md`.
3. Run project commands from `E:\XAI research\solarphysx`, using the env interpreter directly:
   `C:\Users\sahil\anaconda3\envs\solarphysx\python.exe -m ...` (your shell tool may not be
   PowerShell and may not have conda activated; verify the path once with `conda env list`).
4. **Long GPU runs** (anything beyond a few minutes, e.g. 60-epoch training or seed loops): don't
   block on them in a tool call. Either run them in the background and poll the output, or give
   Sahil the exact PowerShell command to run in his VS Code terminal and wait for his results.
5. Resume from **§9 Immediate next steps**. Don't redo finished work (§7 lists it).
6. After any significant result/decision/correction, update the vault (§11) and, if the
   project state changed, this file's §7–§9. Commit and push code/results changes.

---

## 1. People, context, deadlines

- **Researcher:** Sahil Manikshete — M.S. Data Science, University of Michigan–Dearborn;
  research assistant to **Prof. Van Hai Bui** (power systems and energy), who wants journal
  publications.
- **Template paper:** Sahil's first-authored power-transmission fault-detection paper
  (arXiv 2608.23726, submitted to SEGAN). Preferred structure: concrete problem → named pipeline
  → physics-derived features → beats a benchmark → ablation isolating what carries the result.
  Past vault for that paper: `D:\PowerFaultDetection_Vault`.
- **Next advisor meeting: Thursday 2026-09-24.** Goal: show credible results (E1 robust, E2
  preliminary, EDA/cleaning findings) and get decisions (§10).
- **Timeline:** ~13 weeks to **submission** (target ≈ 2026-12-14); review adds 3–6 months.
- **Hardware:** Acer Predator Helios Neo 16 laptop, Windows, **RTX 4060 Laptop GPU (8 GB)** —
  the only machine; runs code, vault, training. Keep it plugged in, PredatorSense Performance
  mode, sleep disabled during long runs.

---

## 2. The research in one page

**Title:** SolarPhysX — Physics-Grounded Explainable Forecasting for Photovoltaic Power Plants.

**One line:** Existing PV-forecasting work produces explanations and evaluates them for
*fluency* (ROUGE/BLEU); we evaluate them for *truth*, using physics as the arbiter, and enforce
it at runtime before an operator sees them.

**Why this gap:** A PRISMA systematic review (Energy and AI, July 2026) screened 951 records,
found **173** studies pairing PV forecasting with explanations (SHAP in 69; 43 in 2024 → 73 in
2025), concluded most explainability claims are aspirational, fidelity evaluation is largely
absent, and only **57.2%** justify their explainer choice.

**Strategy:** do NOT claim unprecedented novelty. Adopt a published system as a reproducible
baseline (**SolarTrans**), challenge it on evaluation protocol and explanation validity, then
build the fix. Concrete numbers to beat, named reference point.

### 2.1 Baseline: SolarTrans
Siddiqa, Rana, Khan, Jeribi, Tahir (2025), *Explaining solar forecasts with generative AI: A
two-stage framework combining transformers and LLMs*, **PLOS ONE 20(9): e0331516**,
doi:10.1371/journal.pone.0331516. Open access. **No code released.** PDF:
`E:\XAI research\journal.pone.0331516.pdf`.

- Pipeline: transformer encoder–decoder forecaster (DC power, 8 steps = 2 h) → Flan-T5-Base
  fine-tuned on outputs of a deterministic rule engine (**Algorithm 1**) → Gradio demo (hosted in a
  Kaggle notebook; **no public URL** — don't claim a "live demo").
- Spec (Tables 1–2): encoder input 48×8 (dc_power, irradiation, ambient_temp, module_temp, hour,
  day, weekday, month); decoder input 8×4 (hour, day, weekday, month); d_model 64, 8 heads,
  4 enc + 4 dec layers, FFN 64→2048→64, dropout 0.1, learnable positional encodings, causal
  decoder mask; head LayerNorm → Linear+ReLU+Dropout → Linear(1). AdamW lr 5e-4, wd 1e-3,
  batch 64, ≤60 epochs with early stopping (Plant 1 curves end ≈ epoch 38). Loss ambiguity:
  Eq 1 says L1, Table 4 says MSE (we use MSE). Table 4 lists search ranges; final config is in
  the text. Per-inverter StandardScaler fit on the full series; forward-fill; `random_split`
  70/10/20 with fixed seed; stride not stated (implied 1).
- Reported (random split, **z-scored units**): Plant 1 MAE 0.0782 / RMSE 0.1760 / **R² 0.9692**;
  Plant 2 0.1544 / 0.4424 / **0.7956**; Combined 0.1105 / 0.3189 / 0.8967. Baselines (combined
  R²): LSTM 0.8806, GRU 0.8674, BiLSTM 0.8638, CNN-LSTM 0.8716.
- Explanations (mean over **100** test samples — an aggregate, not per-sample): ROUGE-1 0.7889,
  ROUGE-2 0.7211, ROUGE-L 0.7759, ROUGE-Lsum 0.7771, BLEU 0.6558. Flan-T5: 3 epochs, batch 8,
  lr 1e-5, wd 0.001.
- **Algorithm 1** (rule engine): night hours → nighttime message; I > 0.9 high, > 0.4 moderate,
  else low; Ta > 32 high, < 22 cool, else normal; Tm > 50 very high, else optimal;
  ΔP = P_end − P_start: > 100 strong increasing, < −100 decreasing, else stable (endpoints only
  — a design flaw).

### 2.2 The three weaknesses (W1–W3) and their status
- **W1 — evaluation leakage.** Random split of stride-1 overlapping windows puts near-duplicates
  on both sides. Also: 22 inverters share one weather sensor, so the same timestamp leaks via
  sibling inverters; scaler fit before the split. **STATUS: CONFIRMED numerically (E1, robust).**
- **W2 — self-refuting attribution.** Their SHAP ranks **day-of-month** top (encoder AND decoder)
  and they call it a finding; over 34 days it's a day index. Month is also a quasi-index.
  **STATUS: CONFIRMED (E1 day-shuffle test).**
- **W3 — explanation infidelity.** Fig 12 sample 2 (I = 0.9, Ta 29.0, Tm 58.5, predicted
  11528.5 → 12034.1, ΔP = +506) → rules say moderate-or-high irradiance (never low), very high
  module temp, strong increasing; published text says **low** irradiance, **moderate** module
  temp, **decreasing** → **three** violations. Fig 13 Gradio example: Ta 11 °C (rule: cool) →
  text "normal"; ΔP +18.5 (rule: stable) → text "strong increasing". Samples 1/4 predict negative
  power (down to −22.5). Sample 3 (6468 → 9964 → 7558) labelled "strong increasing" = rule design
  flaw, not hallucination. Caveat: no Fig 12 output matches Algorithm 1 wording verbatim, so claim
  "outputs contradict the **published** rule logic" (meaning-level). **STATUS: supported by the
  paper's text; numerical audit (E5) not done.**

### 2.3 Framework (five modules)
| # | Module | Function |
|---|---|---|
| M1 | Physics-informed forecaster | Inputs + solar zenith/azimuth, air mass, clear-sky GHI/POA, AOI, cell temp, clear-sky index + variability. Separates deterministic geometry from stochastic clouds |
| M2 | Regime router | Smooth vs ramp, ramp-rate metric applied to the **forecast**, so it fires before the event |
| M3 | Dual attribution | Smooth: horizon-aggregated attribution. Ramp: delta attribution + counterfactual re-forecasting |
| M4 | Runtime verification gate | Physics checks on every output **before display** |
| M5 | Operator interface | Forecast, ramp flags, attribution, verified explanation, explicit flags (Gradio) |

**Gate design (M4 / XAI-4):** the explanation module emits **structured claims, not prose**; the
gate verifies claims (physics lookup, not language parsing); text is rendered only from passing
claims. Tiers: **T1 output bounds** (non-negativity, zero below horizon, capacity ceiling — data
says ≈ 1,400 kW DC per inverter, clear-sky upper bound, ramp-rate limit); **T2 attribution
plausibility** (temperature-coefficient sign, irradiance monotonicity, dominance by meaningless
features — would fire on W2, stability under perturbation); **T3 claim truth** (irradiance label
vs clear-sky index, trend vs actual curve shape, magnitude within physical range, dominant driver
mentioned). Scores → **Physical Consistency Score (PCS)**. Three states: **PASS / FLAG / BLOCK**
(binary gating is useless in volatile conditions). Validation: (a) Algorithm 1 as exact ground
truth, compared at claim level (Flan-T5 paraphrases); (b) synthetic violation injection;
report a **risk–coverage curve**. Defend against Adebayo et al. (2018): multiple attribution
methods, report agreement, PCS attribution-agnostic. Open: thresholds via clear-sky index not
absolute cuts; PCS aggregation rule.

**XAI components:** XAI-1 SHAP (comparison layer, not the contribution) · XAI-2 delta attribution
(attribute forecast *change*, + counterfactual re-forecasting) · XAI-3 PCS · XAI-4 runtime gating ·
XAI-5 fidelity evaluation (deletion/insertion, stability, rule conformance).

**Research questions:** RQ1 does random splitting inflate accuracy, by how much? · RQ2 do
geometry/clear-sky features help once shortcuts are removed? · RQ3 are LLM explanations faithful;
can violations be detected automatically? · RQ4 during ramps, does attributing change beat
attributing level? · RQ5 can a runtime gate cut false explanations, at what coverage cost?

**Contributions:** (1) leakage-corrected benchmark + protocol · (2) physics features + ablation ·
(3) delta attribution for ramps · (4) PCS · (5) three-state gate with risk–coverage validation ·
(6) open-source code + interface (baseline released none).

### 2.4 Experiments
| ID | Experiment | Status |
|---|---|---|
| E1 | SolarTrans random vs chronological split (RQ1, W1/W2) | ✅ robust; rolling origin mostly done |
| E2 | Feature ablation calendar / nocal / physics, chronological (RQ2) | 🟨 19/24 runs; direction positive on P2, not yet significant |
| E3 | Ramp detection F1 / CSI — the **primary accuracy claim** (aggregate RMSE and ramp skill decouple, Nouri et al. 2024) | ⬜ |
| E4 | Delta vs level attribution — **GO/NO-GO: if they agree, XAI-2 is void. Run early.** | ⬜ |
| E5 | PCS audit of baseline explanations (Algorithm 1 ground truth) — needs Flan-T5 retrain | ⬜ |
| E6 | Gated vs ungated generation, risk–coverage | ⬜ |
| E7 | Cross-site transfer to DKASC Alice Springs — now **required** (see §6) | ⬜ |

Metrics: MAE, RMSE, R², skill vs persistence (report z-scored AND kW); ramp F1 + CSI; PCS, gate
precision/recall, risk–coverage, deletion/insertion fidelity.

**Original 13-week plan:** W1–2 data + physics · W3–4 reproduce SolarTrans (E1, E2) · W5 E4
go/no-go · W6–7 Algorithm 1 + Flan-T5 + E5 · W8–9 gate + E6 + E3 · W10 E7 DKASC (now to be pulled
earlier) · W11–12 figures, draft, interface, code release · W13 advisor review + submit.
Write methods/related work in parallel. Cut order if late: E7 → interface polish → extra E3 models.
We are ahead of plan: E1 and most of E2 done in week 2.

---

## 3. Environment

- Project root: `E:\XAI research\solarphysx` (git initialised; `.gitignore`: `/data/`,
  `__pycache__/`, `*.parquet`, `.ipynb_checkpoints/`, `/models/`). **Remote:** private repo
  `solarphysx` on Sahil's **personal** GitHub (branch `main`); commit identity is his personal
  Gmail. Commit and push after each significant result.
- Conda: `C:\Users\sahil\anaconda3`, env **`solarphysx`** (Python 3.11). Packages: torch (CUDA
  build), pandas, pyarrow, numpy, pvlib, shap, scikit-learn, matplotlib, jupyter, ipykernel.
- VS Code: interpreter = solarphysx env; `.vscode/settings.json` sets terminal cwd to workspace,
  autosave, Ruff formatter, `jupyter.notebookFileRoot` = workspace.
- **Shell is Windows PowerShell:** no bash `$()`; change drive with `E:` then `cd`; paths with
  spaces need quotes. Loops: `foreach ($p in 1,2) { ... }`.
- **Run modules from the project root with `python -m src.<pkg>.<module>`** (packages need the
  `__init__.py` files that exist in `src/`, `src/data/`, `src/models/`, `src/experiments/`).
- Other files in `E:\XAI research\`: the SolarTrans PDF, `SolarPhysX-Research-Proposal.pdf`
  (3-page proposal shown to the professor), `SolarPhysX-Meeting-Script.pdf` (walkthrough script),
  `SolarPhysX-Context-Handoff.md` (original handoff), plus unsummarised PDFs `Can we trust XAI.pdf`,
  `Shapely Explanation.pdf`, `thursday-pitch.md.pdf`.

---

## 4. Data

### 4.1 Kaggle "Solar Power Generation Data" (anikannal) — the baseline's own data
`data/raw/Plant_{1,2}_Generation_Data.csv` (DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER,
DAILY_YIELD, TOTAL_YIELD) and `Plant_{1,2}_Weather_Sensor_Data.csv` (DATE_TIME, PLANT_ID,
SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION). Two plants in India, 22
inverters each (one weather sensor per plant), 15-min, 2020-05-15 00:00 → 2020-06-17 23:45
(34 days, 3,264 steps). Date formats: P1 generation `%d-%m-%Y %H:%M`; the other three
`%Y-%m-%d %H:%M:%S`.

### 4.2 Audit + visual EDA findings (all verified)
- **Plant 1 DC is 10× inflated** (median DC/AC 10.22 vs P2 1.022). Fix: ×0.1. DC–AC is perfectly
  linear with zero scatter → **AC is derived from DC** → dropped as a feature. The baseline never
  corrected this (Fig 12 shows 11,000+ "kW" from one inverter). Its z-scored metrics are
  unaffected (per-inverter z-score cancels a constant factor); its kW outputs are not.
- Row counts: P1 68,778 with left join = paper's 68,774 + 4 rows lacking weather. P2 67,698.
  Combined 136,472 = paper.
- **Missing rows, not NaNs:** P1 inverters have 3,104–3,155 of 3,264 steps; P2 as low as 2,355.
  Forward-fill doesn't fix missing rows, so baseline windows silently span gaps: **17% of
  baseline windows span a gap; max span 240 h instead of 14 h.**
- P1: power tracks irradiance, no timestamp offset; logger outages are plant-wide (longest gap
  35 steps on every inverter). Clear day 20 May; cloudy 21–22 May.
- **P2 is equipment-damaged:** power decouples from irradiance (midday plateaus ~700 kW at
  ~1.0 kW/m²); thick DC ≈ 0 smear at high irradiance; **4 inverters dark ≈ 05-20 → 05-28
  (841 steps ≈ 8.8 days)**; per-inverter daily energy scatters 1–10 MWh.
- Max irradiation 1.222 / 1.099 kW/m²; daily 3–7 kWh/m² (units sane). Zero negative DC rows in
  the data → negative values in baseline figures are model outputs (a T1 example).
- The temperature coefficient cannot be read off a DC-vs-irradiance colour plot (temperature and
  irradiance are confounded); use a narrow irradiance slice when building T2.

### 4.3 Cleaning (`src/data/clean_kaggle.py` → `data/interim/clean_plant{1,2}.parquet`)
Full 15-min grid per inverter (71,808 rows per plant); P1 DC ×0.1 with an assert that DC/AC ends
in (0.95, 1.10) (both 1.022); AC, DAILY_YIELD, TOTAL_YIELD dropped; weather rebuilt at plant
level and merged (fixes the loader's left-join loss), interpolated only for gaps ≤ 2 steps (30 min).
Flags: `gen_observed`, `wx_observed`, **`fault`**, `valid` = observed & weather present & not fault.
Fault rules: **dead** = irradiance > 0.2 kW/m² & DC < 1 kW; **lagging** = sibling median
> 50 kW & DC < 20% of sibling median (siblings = same plant, same timestamp).
Results: P1 faults 71 (0.3% of sunny observed rows), valid 68,707. **P2 faults 3,953 (15.6% of
sunny rows)**, widespread (top-5 inverters 8–10% each, none of them the 4 dark ones), valid
63,745. Partial drops (200–800 kW at full sun) are deliberately NOT flagged (can't be separated
from local cloud shade the single sensor misses) — the flag is conservative.
**Important: no model has used the cleaned data yet.** E1/E2 deliberately use the baseline's
uncorrected preprocessing. Planned: report two protocols — all rows (fair vs baseline) and valid
rows (true forecastability).

### 4.4 Plant locations (not published; fitted from data, `src/data/solar_physics.py`)
P1 **11.5°N, 79.3°E** (clear-sky curve fit r = 0.958); P2 **12.0°N, 80.3°E** (r = 0.997). South
India, consistent between plants; solar noon ≈ 12:05 IST. Longitude (timing) is strongly
identified; it's an *effective* longitude that absorbs the unknown timestamp convention (interval
start vs end). Latitude is weakly identified by curve shape (r 0.999 → 0.991 over 0–28°), so it's
fitted from **day length** (observed 48.8 daylight steps vs model 48 at 8°, 50 at 20°); ±~5°,
peak clear-sky changes ~3% across that range. Clear-sky model: **Haurwitz** (zenith only; swap to
Ineichen if turbidity data is added). Visually verified: measured irradiance stays under the
clear-sky ceiling and touches it on clear moments; sunrise/sunset aligned; clear-sky index
0.4–0.6 on cloudy 11–12 June; dawn/dusk csi spikes capped (csi = 0 when clear-sky GHI
< 0.05 kW/m², clipped to [0, 1.5]). State it in methods as an estimated assumption.

### 4.5 Other data
- **DKASC Alice Springs:** multi-year plant output + co-located meteorology, several array
  technologies, free. **Upgraded to required and should run earlier (right after E1/E2):** 34 days
  of one season can't test physics features (see E2 finding) or give enough ramps.
- **Open-Meteo Historical Forecast API / NOAA HRRR:** forecast inputs for day-ahead. ⚠️ Verify
  Open-Meteo returns archived *forecasts*, not reanalysis.

---

## 5. Code map (all tested against the real data)

```
src/
  data/load_kaggle.py        RAW, FORMATS, load_plant(): parse + left-merge (baseline-style)
  data/clean_kaggle.py       our cleaning → data/interim/clean_plant{p}.parquet (§4.3)
  data/solar_physics.py      LOCATIONS, clear_sky(), fit_location(), add_physics() → cos_zenith,
                             azimuth, ghi_cs (kW/m²), csi. `python -m src.data.solar_physics` refits
  data/baseline_windows.py   baseline preprocessing replica (dropna weather, ffill, calendar
                             features, add_physics, per-inverter z-score on FULL series), stride-1
                             windows 48+8; FEATURE_SETS; random_split_idx; chrono_split_idx(frac)
                             with 14 h gap after each cut and test end bound; neighbour_leak_share.
                             `python -m src.data.baseline_windows` = self-check (prints 4 test
                             weeks + "ok"; asserts default split = 91020/9460/23804 combined)
  models/solartrans.py       SolarTrans(n_enc, n_dec, ...) per Tables 1–2
  experiments/e1_leakage.py  one script for E1, E2 and rolling origin (flags below)
  experiments/summarize_e1.py  tables + all figures
notebooks/01_eda_kaggle.ipynb  visual EDA + fault-flag check plots
results/e1/                  one JSON per run + e1_runs.csv, e1_summary.csv and figures
```

**FEATURE_SETS** (encoder, decoder): `calendar` = published (CONT + hour, day, weekday, month /
hour, day, weekday, month); `nocal` = CONT + hour / hour; `physics` = CONT + cos_zenith, ghi_cs,
csi / cos_zenith, ghi_cs (no hour; decoder gets the future clear-sky curve; no azimuth — the
ordered sequence separates AM/PM). CONT = DC_POWER, IRRADIATION, AMBIENT_TEMPERATURE,
MODULE_TEMPERATURE.

**`e1_leakage.py` flags:** `--plant {1,2,combined}` · `--split {random,chrono}` ·
`--features {calendar,nocal,physics}` · `--train-frac` (chrono; default 0.7; 0.4/0.5/0.6 move the
test week earlier) · `--seed` · `--epochs` (60) · `--patience` (8) · `--clip` (0 = off) ·
`--tag` · `--skip-existing` · `--max-batches` (debug).
Output name auto-built: `plant{p}_{split}[_{features}][_origin{NN}][_{tag}][_clip]_s{seed}.json`.
Each JSON stores: config, test_period, counts, neighbour_leak_share, per-epoch val MSE / test R²
(monitoring only, never used for selection) / train loss, best_epoch, clip, and metrics for
`model`, `persistence` (repeat last observed DC over the horizon), `day_shuffled` (only if `day`
is an input). All metrics are in **z-scored units** (as the paper's Table 5).

**Figures from `summarize_e1`:** `e1_summary.png` (paper vs random vs chrono R², skill, day
reliance — unclipped E1 runs, tag ""), `e1_test_curves.png` (per-epoch test R², calendar @ 0.7),
`e1_rolling_origin.png` and `e2_features.png` (clipped runs only, so one protocol).

**Known issues and how they were handled (don't rediscover):**
- Some GPU runs went to **nan in epoch 1** (never on CPU; inputs verified NaN-free). Gradient
  clipping (`--clip 1.0`) did NOT fix all of them → the cause is most likely the
  `nn.TransformerEncoder` **eval-mode fast path**, now disabled via
  `torch.backends.mha.set_fastpath_enabled(False)` (same maths). The script now aborts on
  non-finite loss, saves nothing, and says whether **TRAINING** diverged (→ retry `--clip 0.5`)
  or **EVALUATION** produced nan (→ investigate). ⚠️ The fast-path fix is **unverified on the
  GPU**; 5 runs still fail (§9).
- **Protocol decision:** every E2 and rolling-origin run uses `--clip 1.0` (auto-tagged `_clip`),
  including a re-run of the calendar baseline; finished unclipped E1 runs remain the
  published-protocol reference. Report the clip as a protocol deviation.
- pandas: never use `df.clip` for a column named `clip` (it's a method) → `df["clip"]`.
- A CPU-only sandbox can smoke-test paths with `--epochs 1 --max-batches 5`.

---

## 6. Results so far (z-scored; skill = 1 − RMSE_model / RMSE_persistence)

### 6.1 Model-free facts
- 91% of random-split test windows have a near-duplicate (55/56 shared timesteps) in training;
  chronological 0.2%.
- 17% of baseline windows span timestamp gaps (max 240 h).
- Our preprocessing reproduces the paper's record counts exactly (136,472).

### 6.2 E1 — leakage (robust)
| | Random (their method) | Chronological (honest) |
|---|---|---|
| P1 R² | 0.986 ± 0.001 (paper 0.969) | 0.814 ± 0.023 |
| P2 R² | 0.863 ± 0.001 (paper 0.796) | 0.410 ± 0.075 |
| P1 skill | 0.79 | **0.24 ± 0.05** |
| P2 skill | 0.46 | **−0.12 ± 0.07 (all 3 seeds < 0)** |
| R² lost when day-of-month shuffled | 0.11 / 0.17 | ≈ 0 |

- Reproduction is faithful (random split matches/exceeds the paper).
- **Persistence R² is unchanged across splits** (P1 0.684 vs 0.678; P2 0.531 vs 0.528) → the drop
  is the model losing its leaked advantage, not a harder test week.
- Under-training ruled out: full 60 epochs gives no gain (P1 0.787, P2 0.360); even the oracle
  best epoch (peeking at test) never exceeds R² ≈ 0.85 (P1) / 0.58 (P2); with more training,
  random-split test R² rises while P1 chrono test R² **falls** (memorisation signature).
- W2 confirmed: day-of-month reliance exists only under leakage.
- Headline sentence: *"Under leakage-free evaluation, the published transformer's skill over
  persistence falls from 0.79 to 0.24 ± 0.05 on Plant 1 and becomes negative on Plant 2
  (−0.12 ± 0.07, all seeds)."*

### 6.3 Rolling origin (calendar, clip 1.0, seed 0) — skill per test week
| Test week | P1 | P2 |
|---|---|---|
| 40% origin | missing (fails) | missing (fails) |
| 50% (≈ 4–11 June) | **−0.34** | 0.11 |
| 60% (≈ 8–14 June) | 0.13 | −0.07 |
| 70% (11–17 June, E1 means) | 0.24 | −0.12 |
| Random split | 0.79 | 0.46 |
→ E1 holds on three test weeks; even clean P1 falls below persistence on 4–11 June. Week-to-week
variation is large (single seeds).

### 6.4 E2 — feature ablation (chronological, 70%, clip 1.0)
| Skill | calendar | nocal | physics |
|---|---|---|---|
| **P1** | 0.275 (n=1; E1 unclipped 0.241 ± 0.047, n=3) | 0.246 ± 0.010 (n=3) | 0.244 ± 0.058 (0.187, 0.242, 0.303) |
| **P2** | −0.134 ± 0.183 (+0.055, −0.146, −0.310) | +0.093 (0.134, 0.051; n=2) | +0.095 ± 0.116 (0.021, 0.228, 0.035) |
P2 R²: calendar 0.383 → nocal 0.611 → physics 0.609.
- P1: clear tie.
- P2: removing the calendar index helps **on average** (+0.23 skill, +0.23 R²) but is **not
  significant** with n=3 (paired physics−calendar by seed: −0.034, +0.374, +0.345 → t ≈ 1.7,
  p ≈ 0.2). Needs 5 seeds per condition.
- Physics ≈ nocal on 34 days because `hour` already encodes the sun path when it barely changes
  between days → the value of physics features must be tested on **DKASC (multi-season)**. This
  is a scientific argument for DKASC, not just generalisation.
- **New finding: leakage hides instability.** Random-split seeds agree to ±0.001 R²; under
  honest evaluation P2 calendar skill spans +0.06 to −0.31 across seeds alone.

### 6.5 Status for Thursday
| Finding | Status |
|---|---|
| Leakage inflates accuracy; skill collapses (E1) | ✅ robust (3 seeds, 3 test weeks, under-training ruled out) |
| Day-of-month importance is a leakage artifact (W2) | ✅ |
| Leakage hides seed instability | ✅ |
| Removing calendar helps P2 (E2) | 🟨 positive direction, not significant |
| Physics features beat hour of day | ❌ not on 34 days; needs DKASC |

---

## 7. Finished work (do not redo)
Env + VS Code set up · loader + numeric audit · visual EDA (8-cell notebook) · cleaning + fault
flag (visually validated) · SolarTrans replica · E1 (random/chrono, 3 chrono seeds, 2 random,
full60, oracle bound, day shuffle) · plant location fit + physics features (visually checked) ·
feature sets + rolling-origin split (self-check) · E2 19/24 runs · rolling origin 50/60/70% ·
vault with notes per experiment · W1–W3 audit from the paper text.

## 8. Not started
E3 ramp F1/CSI · E4 delta vs level attribution (GO/NO-GO) · E5 PCS audit (needs Algorithm 1
reimplementation + Flan-T5-Base retrain) · E6 gate + risk–coverage · E7 DKASC · our own pipeline
on cleaned data (valid-rows protocol, train-only scaler, windows skipping invalid rows) · SHAP
comparison layer · Gradio interface · paper writing · ramp-event count (statistical power).

## 9. Immediate next steps (in order)
1. **Diagnose the 5 failing runs** (P1 calendar-clip s0 and s2; P2 nocal s2; both 40% origins).
   Run each alone and read the last lines (TRAINING vs EVALUATION):
   ```powershell
   python -m src.experiments.e1_leakage --plant 1 --split chrono --train-frac 0.4 --clip 1.0 --skip-existing
   python -m src.experiments.e1_leakage --plant 1 --split chrono --seed 0 --clip 1.0 --skip-existing
   python -m src.experiments.e1_leakage --plant 1 --split chrono --seed 2 --clip 1.0 --skip-existing
   python -m src.experiments.e1_leakage --plant 2 --split chrono --features nocal --seed 2 --clip 1.0 --skip-existing
   python -m src.experiments.e1_leakage --plant 2 --split chrono --train-frac 0.4 --clip 1.0 --skip-existing
   ```
   TRAINING → retry with `--clip 0.5`. EVALUATION → the fast-path fix didn't work; investigate
   (e.g. check `torch.__version__`, run `predict` with the model in train mode + `torch.no_grad`,
   or evaluate on CPU). Keep fixes minimal and protocol-neutral.
2. **Settle E2 on Plant 2:** seeds 3–4 for all three feature sets, then summarise:
   ```powershell
   foreach ($s in 3,4) { foreach ($f in "calendar","nocal","physics") {
       python -m src.experiments.e1_leakage --plant 2 --split chrono --features $f --seed $s --clip 1.0 --skip-existing
   } }
   python -m src.experiments.summarize_e1
   ```
   Report mean ± std and a paired test by seed; keep claims to what the stats support.
3. **Meeting prep (Wednesday):** condensed 1–2 page Word brief (Sahil prefers clean, condensed
   Word outputs): figures `e1_summary.png`, `e1_test_curves.png`, `e1_rolling_origin.png`,
   `e2_features.png`; the §6.5 table; the questions in §10. Optionally update the proposal and
   meeting script wording (W3 = three violations; ROUGE 0.7889 is an aggregate; no live demo).
4. **After the meeting:** apply advisor decisions → our pipeline on cleaned data (valid-rows) →
   **E4 go/no-go early** → DKASC loader + E2 on multi-season data → E5 (Algorithm 1 + Flan-T5).

## 10. Questions for Prof. Bui (Thursday)
Horizon: intra-day (2–8 h) or day-ahead (10–24 h; needs NWP, shifts explanations to forecast
uncertainty)? · DKASC as a **required** second dataset, pulled earlier? · Keep the LLM layer with a
verification gate, or verified templates only? · Emphasis: physics verification vs forecast
accuracy? · Target journal (page limits shape the writing)?

---

## 11. Working rules (follow these)

**Sahil's preferences**
- Build incrementally, phase by phase; explain the *why* before the *how*, in plain terms.
- **Visual EDA in a notebook before any cleaning or processing decision.** Don't make data
  decisions from summary numbers alone.
- Windows/PowerShell commands only. Give exact commands to run from the project root.
- Sahil has intermediate development experience. Historically he typed code himself; under the
  meeting deadline Claude wrote files directly. Ask if unsure which he wants for a given task.
- Prefers clean, condensed Word outputs and portable context documents.
- Search for prior art before asserting novelty — several ideas were revised after searches.

**Coding style (ponytail)**
- Simplest thing that works; reuse existing modules and flags before adding files; stdlib/
  installed packages before new dependencies; no speculative abstraction.
- One runnable self-check for non-trivial logic (assert-based `__main__` blocks, as in
  `baseline_windows.py` and `solar_physics.py`).
- Mark deliberate shortcuts with `# ponytail:` comments.
- Test new paths before handing over (CPU smoke run with `--epochs 1 --max-batches 5`).

**Research integrity**
- Never claim a result the data doesn't support; mark unverified claims "pending".
- Every reported number states: dataset/plant, split, feature set, seeds, clip, units (z-scored
  vs kW).
- Keep protocol deviations explicit (clipping, fast-path fix); only one variable changes per
  comparison.
- The baseline reproduction uses the baseline's preprocessing *uncorrected*; our corrections
  live only in our own pipeline.

**Vault protocol (`E:\XAI research\SolarPhysX_Vault`)**
- Significant = a result, decision, corrected claim, finished phase, or new blocker.
- Update in order: topic note (`04_Experiments/…`, `03_Data/Datasets.md`, `02_Baseline/…`) →
  `00_Project_State/Changelog.md` (one dated line, newest first) → `Project State Summary.md`
  ("Where we stopped" + "Immediate next step", < ~60 lines) → `06_Daily_Logs/YYYY-MM-DD.md` →
  `Open Questions & Next Steps.md` if relevant.
- Read a note before editing it; never overwrite wholesale; use today's actual date.
- Key notes: `02_Baseline/SolarTrans.md`, `02_Baseline/Weakness Audit W1-W3.md`,
  `01_Framework/Framework Overview.md`, `01_Framework/Verification Gate.md`,
  `03_Data/Datasets.md`, `04_Experiments/E1 Leakage.md`, `04_Experiments/E2 Features.md`,
  `04_Experiments/Experiment Plan.md`, `05_Literature/Key Papers.md`.

---

## 12. Open to-dos (not yet scheduled)
- Systematic novelty search (IEEE Xplore, Scopus, Web of Science): "ramp explanation",
  "explanation faithfulness PV", "physics-consistent explanation".
- Read the PRISMA review in full; mine its references.
- Differentiate from Two-Stage PV Forecasting (arXiv 2603.04132).
- Verify Open-Meteo archived-forecast semantics.
- Count ramp events in the Kaggle data (power for E3/E4).
- Summarise the three unsummarised PDFs in `E:\XAI research\` if relevant.

## 13. Key papers
| Paper | Role | Link |
|---|---|---|
| Siddiqa et al. 2025, PLOS ONE 20(9) e0331516 | Baseline | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0331516 |
| PRISMA review, Energy and AI, Jul 2026 | Field map + gap | https://www.sciencedirect.com/science/article/pii/S2772940026000275 |
| Nouri et al. 2024, Solar RRL 8:2400468 | Ramp metric; RMSE vs ramp skill decouple (generative model F1 ≥ 0.7) | https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202400468 |
| Frye, Rowat & Feige, ICLR 2021 | On-manifold explanations | https://arxiv.org/abs/2006.01272 |
| Adebayo et al., NeurIPS 2018 | Saliency sanity checks (defend the gate) | https://arxiv.org/abs/1810.03292 |
| Two-Stage PV Forecasting, arXiv 2603.04132 | Close relative — differentiate | https://arxiv.org/abs/2603.04132 |
| Machlev et al. 2022, Energy and AI 9:100169 | XAI in power systems review | doi:10.1016/j.egyai.2022.100169 |
| Liao et al. 2024, Applied Energy 376:124273 | XAI trustworthiness in wind forecasting | doi:10.1016/j.apenergy.2024.124273 |

## 14. Corrections already made (don't reintroduce)
- W3 has **three** violations in Fig 12 sample 2 (not two); ROUGE 0.7889 is a 100-sample mean.
- No public live demo exists.
- Baseline trained with early stopping (~38 epochs on P1), not a fixed 60.
- Robustness entries dated 2026-09-22 (briefly mis-dated 23rd).
- My first nan diagnosis (gradient explosion → clipping) was incomplete; see §5 known issues.
- The first `.gitignore` pattern `data/` would have ignored `src/data/`; it's `/data/`.
