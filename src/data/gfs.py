"""Load the GFS point forecasts saved by fetch_gfs.py and turn them into 3-hour means.

GFS writes DSWRF and TCDC as averages since the last 6-hourly reset:
  fxx % 6 == 3 -> mean over (fxx-3, fxx]      (already a 3-h mean)
  fxx % 6 == 0 -> mean over (fxx-6, fxx]      -> 3-h mean over (fxx-3, fxx] = 2*value - value(fxx-3)
Self-check:  python -m src.data.gfs
"""

import pandas as pd

GFS_CSV = "data/raw/gfs/gfs_0p5_alice_springs.csv"
ACST = pd.Timedelta(hours=9, minutes=30)  # Alice Springs local time (no daylight saving), as in DKASC


def to_3h_means(g: pd.DataFrame, offset: pd.Timedelta = ACST) -> pd.DataFrame:
    g = g.sort_values(["run_utc", "fxx"]).copy()
    prev = g.groupby("run_utc")[["fxx", "dswrf", "tcc"]].shift(1)
    six = g.fxx % 6 == 0
    ok = ~six | (prev.fxx == g.fxx - 3)  # a 6-h value needs its fxx-3 partner from the same run
    for c in ["dswrf", "tcc"]:
        g.loc[six, c] = 2 * g.loc[six, c] - prev.loc[six, c]
    g = g[ok].copy()
    g["dswrf"] = g.dswrf.clip(lower=0)
    g["tcc"] = g.tcc.clip(0, 100)
    g["valid_local"] = g.valid_utc + offset  # end of the 3-h window, local (standard) time
    return g


def load_gfs(path: str = GFS_CSV, offset: pd.Timedelta = ACST) -> pd.DataFrame:
    return to_3h_means(pd.read_csv(path, parse_dates=["run_utc", "valid_utc"]), offset)


if __name__ == "__main__":
    run = pd.Timestamp("2022-01-01")
    true3h = [100.0, 300.0, 500.0, 200.0]  # (12,15] (15,18] (18,21] (21,24]
    raw = pd.DataFrame({"run_utc": run, "fxx": [15, 18, 21, 24],
                        "dswrf": [true3h[0], (true3h[0] + true3h[1]) / 2, true3h[2], (true3h[2] + true3h[3]) / 2],
                        "tcc": [10.0, 20.0, 30.0, 40.0], "t2m": 290.0})
    raw["valid_utc"] = run + pd.to_timedelta(raw.fxx, unit="h")
    out = to_3h_means(raw.drop(index=0))  # drop fxx 15 -> fxx 18 has no partner and must be dropped
    assert list(out.fxx) == [21, 24] and list(out.dswrf) == [500.0, 200.0], out
    assert list(to_3h_means(raw).dswrf) == true3h
    print("ok")
