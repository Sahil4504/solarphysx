"""Fetch archived NOAA GFS forecasts for DKASC (Alice Springs): inputs for the day-ahead model.

One run per day: 00 UTC (published ~03:30-04:30 UTC, i.e. ~13:00-14:00 local ACST), forecast hours
15-36 every 3 h = valid 00:30-21:30 ACST the NEXT day. An operator issuing at ~18:00 ACST forecasts
next-day daylight 12-25 h ahead using only data that existed at issue time (no leakage).

Archive: AWS (with Google fallback), which starts 2021-01-01. Earlier forecasts are not usable: the NCEI
archive has no 2018-2020 forecast files at Herbie's address, and NCAR holds only full ~0.5 GB files.
0.5 deg grid (smaller downloads than 0.25 deg). Values are bilinearly interpolated to the site. Stored raw, as GFS writes them:
  dswrf, tcc : interval AVERAGES over (fxx-3, fxx] if fxx % 6 == 3, over (fxx-6, fxx] if fxx % 6 == 0
               -> de-average in the feature step (that is why fxx 15 is fetched).
  t2m        : instantaneous, kelvin.
Resumable: (run, fxx) pairs already in the CSV are skipped; failures go to *_failed.csv.

Usage (project root):
  python -m src.data.fetch_gfs --start 2022-06-01 --end 2022-06-01 --workers 1   # test
  python -m src.data.fetch_gfs --start 2021-01-01 --end 2024-02-05               # full run (resumable)
"""

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from herbie import Herbie

warnings.filterwarnings("ignore")  # module level so worker processes are quiet too

LAT, LON = -23.762, 133.875  # ponytail: DKASC approx.; verify against the data like the Kaggle location fit
FXX = range(15, 37, 3)
SEARCH = ":(?:DSWRF:surface|TCDC:entire atmosphere[^:]*|TMP:2 m above ground):"  # NCEI may append "(considered as a single layer)"
OUT = Path("data/raw/gfs/gfs_0p5_alice_springs.csv")
FAILED = OUT.with_name(OUT.stem + "_failed.csv")
COLS = ["run_utc", "fxx", "valid_utc", "dswrf", "tcc", "t2m", "source"]
NAMES = {"sdswrf": "dswrf", "dswrf": "dswrf", "tcc": "tcc", "t2m": "t2m"}  # cfgrib short names


def fetch(run: pd.Timestamp, fxx: int) -> list:
    H = Herbie(run, model="gfs", product="pgrb2.0p50", fxx=fxx,
               priority=["aws", "google"], verbose=False)
    dss = H.xarray(SEARCH, remove_grib=True, overwrite=True)  # never reuse a half-downloaded file
    vals = {}
    for ds in dss if isinstance(dss, list) else [dss]:
        point = ds.interp(latitude=LAT, longitude=LON)
        vals.update({NAMES[k]: float(point[k]) for k in point.data_vars if k in NAMES})
    # validation at the trust boundary: a wrong message or grid would show up here
    assert set(vals) == {"dswrf", "tcc", "t2m"}, f"missing variables: {sorted(vals)}"
    assert 0 <= vals["dswrf"] <= 1400 and 0 <= vals["tcc"] <= 100 and 220 < vals["t2m"] < 335, vals
    valid = run + pd.Timedelta(hours=fxx)
    return [run, fxx, valid, vals["dswrf"], vals["tcc"], vals["t2m"], H.grib_source]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert pd.Timestamp(args.start) >= pd.Timestamp("2021-01-01"), "GFS archive on AWS starts 2021-01-01"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, parse_dates=["run_utc"])
        done = set(zip(prev.run_utc, prev.fxx))
    jobs = [(run, f) for run in pd.date_range(args.start, args.end, freq="D")
            for f in FXX if (run, f) not in done]
    print(f"{len(done)} rows already saved, {len(jobs)} to fetch")

    new_file = not OUT.exists()
    with open(OUT, "a", newline="") as fo, open(FAILED, "a", newline="") as ff, \
            ProcessPoolExecutor(args.workers) as pool:  # processes: eccodes is not thread-safe
        out, bad = csv.writer(fo), csv.writer(ff)
        if new_file:
            out.writerow(COLS)
        futures = {pool.submit(fetch, run, f): (run, f) for run, f in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            run, f = futures[fut]
            try:
                out.writerow(fut.result())
            except Exception as e:  # keep going; failures are logged and retried on the next run
                bad.writerow([run, f, type(e).__name__, str(e)[:200]])
            if i % 200 == 0 or i == len(jobs):
                fo.flush(); ff.flush()
                print(f"{i}/{len(jobs)} done")


if __name__ == "__main__":
    main()
