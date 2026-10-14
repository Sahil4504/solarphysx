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

US transfer sites (PVDAQ 1423 Henderson NV, 50 Golden CO): issue 18:00 local standard time on D-1; the newest run online
then is the 18 UTC run of D-1 (~13:30-14:30 local), whose fxx 15-36 cover day D 04-22 local. Both sites share each download.

Usage (project root):
  python -m src.data.fetch_gfs --start 2022-06-01 --end 2022-06-01 --workers 1   # test (Alice Springs)
  python -m src.data.fetch_gfs --start 2021-01-01 --end 2024-02-05               # full run (resumable)
  python -m src.data.fetch_gfs --sites henderson golden --start 2022-01-25 --end 2023-02-28
"""

import argparse
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from herbie import Herbie

warnings.filterwarnings("ignore")  # module level so worker processes are quiet too

# name: (lat, lon, GFS run hour UTC used for an 18:00-local issue)
SITES = {"alice_springs": (-23.762, 133.875, 0), "henderson": (36.0275, -114.9215, 18), "golden": (39.742, -105.1727, 18)}
FXX = range(15, 37, 3)
SEARCH = ":(?:DSWRF:surface|TCDC:entire atmosphere[^:]*|TMP:2 m above ground):"  # NCEI may append "(considered as a single layer)"
out_path = lambda site: Path(f"data/raw/gfs/gfs_0p5_{site}.csv")
COLS = ["run_utc", "fxx", "valid_utc", "dswrf", "tcc", "t2m", "source"]
NAMES = {"sdswrf": "dswrf", "dswrf": "dswrf", "tcc": "tcc", "t2m": "t2m"}  # cfgrib short names


def fetch(run: pd.Timestamp, fxx: int, sites: tuple) -> dict:
    H = Herbie(run, model="gfs", product="pgrb2.0p50", fxx=fxx,
               priority=["aws", "google"], verbose=False)
    dss = H.xarray(SEARCH, remove_grib=True, overwrite=True)  # never reuse a half-downloaded file
    rows = {}
    for site in sites:
        lat, lon, _ = SITES[site]
        vals = {}
        for ds in dss if isinstance(dss, list) else [dss]:
            point = ds.interp(latitude=lat, longitude=lon % 360)  # GFS longitudes run 0-360
            vals.update({NAMES[k]: float(point[k]) for k in point.data_vars if k in NAMES})
        # validation at the trust boundary: a wrong message or grid would show up here
        assert set(vals) == {"dswrf", "tcc", "t2m"}, f"missing variables: {sorted(vals)}"
        assert 0 <= vals["dswrf"] <= 1400 and 0 <= vals["tcc"] <= 100 and 220 < vals["t2m"] < 335, vals
        rows[site] = [run, fxx, run + pd.Timedelta(hours=fxx), vals["dswrf"], vals["tcc"], vals["t2m"], H.grib_source]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sites", nargs="*", default=["alice_springs"], choices=list(SITES))
    args = ap.parse_args()
    assert pd.Timestamp(args.start) >= pd.Timestamp("2021-01-01"), "GFS archive on AWS starts 2021-01-01"
    hours = {SITES[s][2] for s in args.sites}
    assert len(hours) == 1, "sites fetched together must use the same GFS run hour"
    run_hour = hours.pop()

    outs = {s: out_path(s) for s in args.sites}
    outs[args.sites[0]].parent.mkdir(parents=True, exist_ok=True)
    done = None  # a (run, fxx) is skipped only when every requested site already has it
    for s, o in outs.items():
        prev = pd.read_csv(o, parse_dates=["run_utc"]) if o.exists() and o.stat().st_size else None
        have = set(zip(prev.run_utc, prev.fxx)) if prev is not None else set()
        done = have if done is None else done & have
    runs = pd.date_range(args.start, args.end, freq="D") + pd.Timedelta(hours=run_hour)
    jobs = [(run, f) for run in runs for f in FXX if (run, f) not in done]
    print(f"{len(done)} (run, fxx) already saved for all sites, {len(jobs)} to fetch")

    files = {s: open(o, "a", newline="") for s, o in outs.items()}
    new = {s: o.stat().st_size == 0 for s, o in outs.items()}
    failed = outs[args.sites[0]].with_name("gfs_0p5_" + "_".join(args.sites) + "_failed.csv")
    with open(failed, "a", newline="") as ff, ProcessPoolExecutor(args.workers) as pool:  # processes: eccodes is not thread-safe
        writers, bad = {s: csv.writer(f) for s, f in files.items()}, csv.writer(ff)
        for s in args.sites:
            if new[s]:
                writers[s].writerow(COLS)
        futures = {pool.submit(fetch, run, f, tuple(args.sites)): (run, f) for run, f in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            run, f = futures[fut]
            try:
                for s, row in fut.result().items():
                    writers[s].writerow(row)
            except Exception as e:  # keep going; failures are logged and retried on the next run
                bad.writerow([run, f, type(e).__name__, str(e)[:200]])
            if i % 200 == 0 or i == len(jobs):
                for fo in files.values():
                    fo.flush()
                ff.flush()
                print(f"{i}/{len(jobs)} done", flush=True)
    for fo in files.values():
        fo.close()


if __name__ == "__main__":
    main()
