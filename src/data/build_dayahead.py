"""Day-ahead table for DKASC #38: one row per (target day D, local hour h), hours 05-19 ACST.

Issue time: 18:00 ACST on D-1. Allowed inputs then:
  - GFS 00 UTC run of D-1 (issued 09:30 ACST, online by ~14:00 ACST), leads 15-36 h -> valid on day D
  - clear-sky physics for D (deterministic, known in advance)
  - measured power up to 18:00 on D-1 (hours >= 18 fall back to D-2)
Target: mean AC power (kW) over [h, h+1), needs >= 10 of 12 valid 5-min samples.
Output: data/processed/dayahead38.parquet.   Run + self-check:  python -m src.data.build_dayahead
"""

import numpy as np
import pandas as pd
import pvlib

from src.data.clean_dkasc import CAP, LAT, LON, TZ
from src.data.gfs import load_gfs

CLEAN, OUT = "data/interim/dkasc38_clean.parquet", "data/processed/dayahead38.parquet"
TILT, AZ = 14.5, 354.0  # fitted in clean_dkasc
GAMMA = -0.0037  # 1/K, typical mono-Si temperature coefficient (Q CELLS datasheet range -0.34..-0.39 %/K)
HOURS = range(5, 20)
ISSUE_HOUR = 18
SPLIT = {"train": ("2021-01-01", "2022-09-30"), "val": ("2022-10-01", "2022-12-31"), "test": ("2023-01-01", "2023-09-30")}
LOC = pvlib.location.Location(LAT, LON, tz=TZ)


def pv_unit(index, ghi, t_air, wind=1.0):
    """kW from GHI for derate 1.0 (Erbs split -> Hay-Davies POA -> SAPM cell temp -> PVWatts-style DC)."""
    ghi, t_air = np.asarray(ghi, float), np.asarray(t_air, float)
    sp = LOC.get_solarposition(index)
    sp.index = index
    erbs = pvlib.irradiance.erbs(ghi, sp.zenith, index.dayofyear)
    poa = pvlib.irradiance.get_total_irradiance(TILT, AZ, sp.apparent_zenith, sp.azimuth, erbs["dni"], ghi, erbs["dhi"],
                                                dni_extra=pvlib.irradiance.get_extra_radiation(index.dayofyear),
                                                model="haydavies")["poa_global"].fillna(0).clip(lower=0)
    tc = pvlib.temperature.sapm_cell(poa, t_air, wind, -3.56, -0.075, 3)  # open rack
    return poa, (CAP * poa / 1000 * (1 + GAMMA * (tc - 25))).clip(lower=0), sp.apparent_zenith


def clearsky_5min(start, end):
    """5-min Ineichen clear-sky GHI (local, naive index)."""
    idx = pd.date_range(start, end, freq="5min", tz=TZ)
    cs = LOC.get_clearsky(idx, model="ineichen").ghi
    cs.index = cs.index.tz_localize(None)
    return cs


def gfs_hourly(cs5, cs_h):
    """GFS 3-h means -> hourly: interpolate clear-sky index k (not GHI) between window midpoints, rescale by hourly clear-sky."""
    g = load_gfs()
    g["day"] = (g.run_utc + pd.Timedelta(hours=9.5)).dt.normalize() + pd.Timedelta(days=1)  # run on D-1 serves day D
    g["mid"] = g.valid_local - pd.Timedelta(hours=1.5)
    # clear-sky mean over exactly the GFS window (T-3h, T]; 5-min samples are instants
    cs3 = cs5.rolling("3h", closed="right").mean()
    g["cs3"] = cs3.reindex(g.valid_local).to_numpy()
    g["k"] = (g.dswrf / g.cs3).where(g.cs3 > 30).clip(0, 1.2)
    rows = []
    for day, d in g.groupby("day"):
        hrs = pd.date_range(day + pd.Timedelta(hours=HOURS[0]), periods=len(HOURS), freq="h")
        s = d.set_index("mid")[["k", "tcc", "t2m"]].sort_index()
        both = s.index.union(hrs + pd.Timedelta(minutes=30))
        s = s.reindex(both).interpolate(method="time", limit_direction="both")  # shortcut: flat k at the ends of the day
        s = s.reindex(hrs + pd.Timedelta(minutes=30)).set_axis(hrs)
        rows.append(s.assign(day=day))
    h = pd.concat(rows)
    h["t2m"] -= 273.15
    h["ghi_gfs"] = h.k * cs_h.reindex(h.index).to_numpy()
    return h


def build():
    clean = pd.read_parquet(CLEAN).loc["2020-12-25":]
    p = clean.power.where(clean.valid)
    target = p.resample("h").mean().where(p.resample("h").count() >= 10)
    cs5 = clearsky_5min("2020-12-25", "2023-10-01 23:55")
    cs_h = cs5.resample("h").mean()  # labelled by hour start = mean over [h, h+1)

    df = gfs_hourly(cs5, cs_h)
    df["cs_ghi"] = cs_h.reindex(df.index).to_numpy()
    mid = df.index + pd.Timedelta(minutes=30)
    df["poa_gfs"], df["p_phys1"], _ = [x.to_numpy() for x in pv_unit(mid.tz_localize(TZ), df.ghi_gfs, df.t2m)]
    df["poa_cs"], df["p_cs1"], df["zenith"] = [x.to_numpy() for x in pv_unit(mid.tz_localize(TZ), df.cs_ghi, df.t2m)]
    df["hour"] = df.index.hour  # diurnal position, not a calendar shortcut: zenith already carries it
    df["y"] = target.reindex(df.index).to_numpy()

    # measured history as of 18:00 on D-1 (hours >= 18 from D-2)
    lag_days = np.where(df.hour < ISSUE_HOUR, 1, 2)
    lag_idx = df.index - pd.to_timedelta(lag_days, unit="D")
    df["p_lag"] = target.reindex(lag_idx).to_numpy()
    df["pcs_lag"] = df.p_cs1.reindex(lag_idx).to_numpy()  # clear-sky power at the lag time
    seen = df[df.hour < ISSUE_HOUR]
    kd = (seen.y.groupby(seen.day).sum(min_count=8) / seen.p_cs1.groupby(seen.day).sum()).clip(0, 1.2)
    df["kd_lag"] = kd.reindex(df.day - pd.Timedelta(days=1)).to_numpy()  # yesterday's daily clear-sky index

    df["split"] = None
    for name, (a, b) in SPLIT.items():
        df.loc[a:b, "split"] = name
    df = df[df.split.notna()]
    df.index.name = "time"
    return df


if __name__ == "__main__":
    df = build()
    day = df[df.zenith < 85]
    print(df.shape, "days:", df.day.nunique())
    print(day.groupby("split").agg(rows=("y", "size"), y_ok=("y", lambda s: s.notna().mean()),
                                   gfs_ok=("ghi_gfs", lambda s: s.notna().mean())).round(3))
    r = day[["y", "p_phys1", "p_lag"]].dropna()
    print("corr(y, gfs physics) %.3f  corr(y, yesterday) %.3f" % (r.y.corr(r.p_phys1), r.y.corr(r.p_lag)))
    # self-checks: no lookahead in the lag, clear-sky >= GFS on average, physics in the right scale
    t = df.index[df.hour.eq(18)][5]
    assert df.loc[t, "p_lag"] == df.y.get(t - pd.Timedelta(days=2)) or np.isnan(df.loc[t, "p_lag"])
    assert (day.poa_cs.mean() >= day.poa_gfs.mean()) and 0.6 < (r.y.sum() / r.p_phys1.sum()) < 1.0
    assert df.loc["2023-01-01":, "split"].eq("test").all() and not df.loc[:"2022-12-31", "split"].eq("test").any()
    dk = day.y.groupby(day.day).sum() / day.p_cs1.groupby(day.day).sum()
    c = day[day.day.isin(dk[dk > 0.88].index) & day.hour.between(8, 16)]
    kh = c.k.groupby(c.hour).median()
    print("clear-day median GFS k by hour:", kh.round(2).to_dict())
    assert kh.max() - kh.min() < 0.2, "GFS clearness drifts through the day: timing misalignment"
    df.to_parquet(OUT)
    print("saved ->", OUT)
