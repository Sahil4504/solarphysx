"""Replicates the SolarTrans preprocessing (Siddiqa et al. 2025, PLOS ONE) as the paper describes it.

This module deliberately reproduces the baseline's choices, including the ones we criticise,
so that E1 isolates ONE variable: how the windows are split.

Reproduced as published:
- raw Kaggle units (Plant 1 DC left 10x inflated; per-inverter z-scoring cancels this)
- rows missing weather dropped (gives the paper's 68,774 / 67,698 record counts)
- forward-fill of continuous features within each inverter
- 8 encoder features: DC power, irradiation, ambient temp, module temp, hour, day, weekday, month
- 4 decoder features: hour, day, weekday, month (known future time covariates)
- per-inverter z-score of every feature, fit on the full series (before splitting)
- sliding windows of 48 encoder + 8 decoder steps, stride 1, over the rows that exist
  (so windows silently span timestamp gaps, as in the original)

Two split protocols:
- random: 70/10/20 random permutation of windows (the baseline's torch random_split)
- chrono: 70/10/20 of the TIMELINE, cut across all inverters at once; a window belongs to a
  set only if all 56 of its timestamps fall inside that set's period, so windows straddling a
  boundary are discarded. This leaves a gap of at least one window length between sets.

Self-check:  python -m src.data.baseline_windows
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data.load_kaggle import load_plant
from src.data.solar_physics import add_physics

ENC_LEN, DEC_LEN = 48, 8
CONT = ["DC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
TIME = ["hour", "day", "weekday", "month"]
ENC_FEATS = CONT + TIME

# (encoder features, decoder features). Only the features change between sets (E2 ablation).
# ponytail: no azimuth; cos_zenith is symmetric around noon but the ordered ghi_cs sequence
# tells morning from afternoon. Add sin/cos azimuth if attributions show AM/PM confusion.
FEATURE_SETS = {
    "calendar": (ENC_FEATS, TIME),  # baseline, as published
    "nocal": (CONT + ["hour"], ["hour"]),  # drop the day/weekday/month index shortcuts
    "physics": (CONT + ["cos_zenith", "ghi_cs", "csi"], ["cos_zenith", "ghi_cs"]),
}


def prepare_frame(plants: list[int]) -> pd.DataFrame:
    frames = []
    for p in plants:
        df = load_plant(str(p))
        df = df.dropna(subset=["IRRADIATION"])  # paper: malformed entries discarded
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["plant", "SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    df[CONT] = df.groupby("SOURCE_KEY")[CONT].ffill()
    ts = df["DATE_TIME"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["weekday"] = ts.dt.weekday
    df["month"] = ts.dt.month
    return add_physics(df)


def zscore_per_inverter(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or ENC_FEATS
    g = df.groupby("SOURCE_KEY")[cols]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, 1.0)
    return ((df[cols] - mu) / sd).astype("float32")


def build_windows(df: pd.DataFrame, z: pd.DataFrame, enc=None, dec=None) -> dict:
    enc, dec = enc or ENC_FEATS, dec or TIME
    L = ENC_LEN + DEC_LEN
    enc_all, dec_all = z[enc].to_numpy(), z[dec].to_numpy()
    y_all = z["DC_POWER"].to_numpy()
    times_all = df["DATE_TIME"].to_numpy()
    X, T, Y, t_start, t_end = [], [], [], [], []
    for _, idx in df.groupby("SOURCE_KEY").indices.items():
        if len(idx) < L:
            continue
        times = times_all[idx]
        X.append(sliding_window_view(enc_all[idx], (L, len(enc)))[:, 0, :ENC_LEN])
        T.append(sliding_window_view(dec_all[idx], (L, len(dec)))[:, 0, ENC_LEN:])  # known future
        Y.append(sliding_window_view(y_all[idx], L)[:, ENC_LEN:])  # DC power over the horizon
        n_win = len(idx) - L + 1
        t_start.append(times[:n_win])
        t_end.append(times[L - 1 :])
    return {
        "X": np.concatenate(X).astype(np.float32),
        "T": np.concatenate(T).astype(np.float32),
        "Y": np.concatenate(Y).astype(np.float32),
        "t_start": np.concatenate(t_start),
        "t_end": np.concatenate(t_end),
    }


def random_split_idx(n: int, seed: int = 0, frac=(0.7, 0.1, 0.2)):
    perm = np.random.default_rng(seed).permutation(n)
    a = int(frac[0] * n)
    b = a + int(frac[1] * n)
    return perm[:a], perm[a:b], perm[b:]


def chrono_split_idx(t_start: np.ndarray, t_end: np.ndarray, frac=(0.7, 0.1, 0.2)):
    """Cut the timeline at train | val | test fractions, leaving a gap of one full window
    (56 steps = 14 h) after each cut, so no set's inputs sit right next to another set's targets.
    A smaller train fraction moves the test week earlier (rolling-origin evaluation)."""
    gap = pd.Timedelta(minutes=15 * (ENC_LEN + DEC_LEN))
    t0, t1 = pd.Timestamp(t_start.min()), pd.Timestamp(t_end.max())
    c1 = t0 + (t1 - t0) * frac[0]
    c2 = t0 + (t1 - t0) * (frac[0] + frac[1])
    c3 = min(t0 + (t1 - t0) * sum(frac), t1)
    train = np.where(t_end < np.datetime64(c1))[0]
    val = np.where((t_start >= np.datetime64(c1 + gap)) & (t_end < np.datetime64(c2)))[0]
    test = np.where((t_start >= np.datetime64(c2 + gap)) & (t_end <= np.datetime64(c3)))[0]
    return train, val, test


def neighbour_leak_share(train_idx: np.ndarray, test_idx: np.ndarray, n: int) -> float:
    """Share of test windows whose immediate neighbour (index +/-1, i.e. the same inverter
    shifted one step, sharing 55 of 56 timesteps) is in the training set."""
    in_train = np.zeros(n, dtype=bool)
    in_train[train_idx] = True
    prev = np.clip(test_idx - 1, 0, n - 1)
    nxt = np.clip(test_idx + 1, 0, n - 1)
    return float((in_train[prev] | in_train[nxt]).mean())


if __name__ == "__main__":
    # Self-check: default split unchanged from E1, sets separated in time, rolling origins valid.
    df = prepare_frame([1, 2])
    for name, (enc, dec) in FEATURE_SETS.items():
        w = build_windows(df, zscore_per_inverter(df, enc), enc, dec)
        assert w["X"].shape == (134052, ENC_LEN, len(enc)) and w["T"].shape[2] == len(dec), name
        assert not np.isnan(w["X"]).any(), f"{name}: NaN in inputs"
    gap = np.timedelta64(14, "h")
    for tf in (0.4, 0.5, 0.6, 0.7):
        tr, va, te = chrono_split_idx(w["t_start"], w["t_end"], (tf, 0.1, 0.2))
        assert w["t_end"][tr].max() + gap <= w["t_start"][va].min(), tf
        assert w["t_end"][va].max() + gap <= w["t_start"][te].min(), tf
        print(f"train_frac {tf}: train {len(tr)} val {len(va)} test {len(te)} | test "
              f"{str(w['t_start'][te].min())[:10]} -> {str(w['t_end'][te].max())[:10]}")
    assert (len(tr), len(va), len(te)) == (91020, 9460, 23804), "default split changed"
    print("ok")
