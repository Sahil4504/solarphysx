"""Day-ahead table, one row per (target day D, local hour h), hours 05-19 local standard time.
Sites: dkasc38 (training site) and the unseen test sites henderson (PVDAQ 1423), golden (PVDAQ 50) and elb (UM-Dearborn roof).
Power everywhere is in DKASC-equivalent kW (site kW x 5.9 / site kWp), so the DKASC-trained models apply unchanged.

Issue time: 18:00 ACST on D-1. Allowed inputs then:
  - GFS 00 UTC run of D-1 (issued 09:30 ACST, online by ~14:00 ACST), leads 15-36 h -> valid on day D
  - clear-sky physics for D (deterministic, known in advance)
  - measured power up to 18:00 on D-1 (hours >= 18 fall back to D-2)
Target: mean AC power (kW) over [h, h+1), needs >= 10 of 12 valid 5-min samples.
US sites: GFS 18 UTC run of D-1 (online ~13:30-14:30 local) for the same 18:00-local issue; target from 1-min PVDAQ data
(hourly mean, >= 50 of 60 minutes). Altitude is ignored for clear-sky everywhere, as for DKASC, so train and test physics match.
Output: data/processed/dayahead38.parquet (dayahead_<site>.parquet for US).   Run + self-check:  python -m src.data.build_dayahead [site]
"""

import numpy as np
import pandas as pd
import pvlib

from src.data.clean_dkasc import CAP, LAT, LON, TZ
from src.data.gfs import load_gfs

CLEAN = "data/interim/dkasc38_clean.parquet"
SITES = {  # tilt/az: DKASC fitted in clean_dkasc; US from PVDAQ metadata (Henderson confirmed in notebooks/03_eda_pvdaq)
    "dkasc38": dict(lat=LAT, lon=LON, tz=TZ, tilt=14.5, az=354.0, kwp=CAP, gfs="data/raw/gfs/gfs_0p5_alice_springs.csv",
                    utc=9.5, span=("2020-12-25", "2023-10-01 23:55"), out="data/processed/dayahead38.parquet"),
    "henderson": dict(lat=36.0275, lon=-114.9215, tz="Etc/GMT+8", tilt=35.0, az=180.0, kwp=6.48, pvdaq=1423,
                      gfs="data/raw/gfs/gfs_0p5_henderson.csv", utc=-8, span=("2022-01-24", "2023-03-02 23:55"),
                      out="data/processed/dayahead_henderson.parquet"),
    "golden": dict(lat=39.742, lon=-105.1727, tz="Etc/GMT+7", tilt=45.0, az=158.0, kwp=6.0, pvdaq=50,
                   gfs="data/raw/gfs/gfs_0p5_golden.csv", utc=-7, span=("2022-01-24", "2023-03-02 23:55"),
                   out="data/processed/dayahead_golden.parquet"),
    # ELB: no nameplate exists; Prof. Bui (2026-10-19): capacity = maximum hourly output in the data (38.9 kW, 10 May 2026 11:00).
    # Geometry unknown (Facilities has no details) -> flat-roof prior (10 deg, south), not fitted:
    # notebooks/04_eda_elb shows tilt/azimuth are only identifiable from test-window clear days. Snow covers the array
    # Dec-Feb and output per unit sunlight stays low into March (04_eda_elb sec. 5), so calibration = the first month
    # after that (22 Mar - 21 Apr) for the calibrated variant; test = the rest.
    "elb": dict(lat=42.3175, lon=-83.2300, tz="Etc/GMT+5", tilt=10.0, az=180.0, kwp=38.9,
                elb="data/raw/ELB data/ELB Solar Hourly.xlsx", gfs="data/raw/gfs/gfs_0p5_elb.csv", utc=-5,
                span=("2025-11-20", "2026-07-17 23:55"), out="data/processed/dayahead_elb.parquet",
                split={"cal": ("2026-03-22", "2026-04-21"), "test": ("2026-04-22", "2026-07-16")}),
}
US_SPLIT = {"test": ("2022-02-01", "2023-01-31")}  # zero-shot: nothing from the US sites is used for training
GAMMA = -0.0037  # 1/K, typical mono-Si temperature coefficient (Q CELLS datasheet range -0.34..-0.39 %/K)
HOURS = range(5, 20)
ISSUE_HOUR = 18
SPLIT = {"train": ("2021-01-01", "2022-09-30"), "val": ("2022-10-01", "2022-12-31"), "test": ("2023-01-01", "2023-09-30")}
SITE, TILT, AZ, LOC = "dkasc38", 14.5, 354.0, pvlib.location.Location(LAT, LON, tz=TZ)


def use_site(name: str):
    """Point the physics chain (pv_unit, clear-sky, GFS loading; also the gate's counterfactuals) at one site."""
    global SITE, TZ, TILT, AZ, LOC
    s = SITES[name]
    SITE, TZ, TILT, AZ = name, s["tz"], s["tilt"], s["az"]
    LOC = pvlib.location.Location(s["lat"], s["lon"], tz=TZ)


def pv_unit(index, ghi, t_air, wind=1.0):
    """DKASC-equivalent kW from GHI for derate 1.0 (Erbs split -> Hay-Davies POA -> SAPM cell temp -> PVWatts-style DC)."""
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
    s = SITES[SITE]
    g = load_gfs(s["gfs"], pd.Timedelta(hours=s["utc"]))
    g["day"] = (g.run_utc + pd.Timedelta(hours=s["utc"])).dt.normalize() + pd.Timedelta(days=1)  # run on D-1 serves day D
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


CAL_END = "2022-02-01"  # capacity calibration uses only data before the US test window


def pvdaq_hourly(s: dict) -> pd.DataFrame:
    return pd.read_csv(f"data/interim/pvdaq_{s['pvdaq']}_hourly.csv", index_col=0, parse_dates=True)  # from src.data.pvdaq


ELB_SPIKE_KWH = 45.0  # array peaks near 39 kWh/h; larger hourly values follow data gaps (meter catch-up), 04_eda_elb sec. 4


def elb_hourly(s: dict) -> pd.Series:
    """ELB meter, kWh per hour = mean kW over [h, h+1). Raw labels are hour-start in Detroit clock time (DST-aware:
    8 Mar 2026 has 23 rows) -> converted to standard time UTC-5 like every other site."""
    h = pd.read_excel(s["elb"], usecols=[0, 3])
    h.columns = ["ts", "kwh"]
    t = pd.DatetimeIndex(h.ts).tz_localize("America/Detroit", nonexistent="NaT", ambiguous="NaT")
    kw = pd.Series(h.kwh.to_numpy(), index=t)
    kw = kw[kw.index.notna()].tz_convert(s["tz"]).tz_localize(None)
    return kw.where(kw <= ELB_SPIKE_KWH).clip(lower=0).dropna()  # night standby (-0.05) -> 0


def history_rows(site: str, kwp: float = None) -> pd.DataFrame:
    """Hourly measured power with clear-sky physics before the US test window (no GFS, on-site air temperature):
    the 'commissioning' data used for capacity calibration and for the gate's T1 bounds at that site.
    ELB has no on-site air temperature: its calibration rows come from the day-ahead table's 'cal' split (GFS t2m)."""
    use_site(site)
    s = SITES[site]
    if "elb" in s:
        d = build(site, kwp)
        return d[d.split == "cal"][["y", "p_cs1", "cs_ghi", "zenith", "day"]]
    h = pvdaq_hourly(s).loc[:pd.Timestamp(CAL_END) - pd.Timedelta(hours=1)]
    h = h[(h.n_min >= 50) & h.t_amb.notna()]
    cs = clearsky_5min(h.index.min(), h.index.max() + pd.Timedelta(minutes=55)).resample("h").mean().reindex(h.index)
    _, pcs, zen = pv_unit((h.index + pd.Timedelta(minutes=30)).tz_localize(TZ), cs, h.t_amb)
    return pd.DataFrame({"y": h.p_w / 1000 * CAP / (kwp or s["kwp"]), "p_cs1": pcs.to_numpy(), "cs_ghi": cs.to_numpy(),
                         "zenith": zen.to_numpy(), "day": h.index.normalize()}, index=h.index)


def effective_kwp(site: str) -> float:
    """Nameplate scaled so the site's clear-day level matches DKASC's (95th pct of daily measured / clear-sky physics).
    Uses only pre-test data, clear-sky physics and on-site air temperature: no GFS, no test days, no retraining."""
    h = history_rows(site)
    h = h[h.zenith < 85]
    ratio = h.y.groupby(h.day).sum() / h.p_cs1.groupby(h.day).sum()
    d = pd.read_parquet(SITES["dkasc38"]["out"])
    d = d[(d.split == "train") & (d.zenith < 85)].dropna(subset=["y"])
    ref = (d.y.groupby(d.day).sum() / d.p_cs1.groupby(d.day).sum()).quantile(0.95)
    return (SITES[site]["kwp"] or 1.0) * ratio.quantile(0.95) / ref  # no nameplate: y is per kW, so the base is 1


def target_hourly(kwp: float = None) -> pd.Series:
    """Measured mean AC power over [h, h+1) in DKASC-equivalent kW; NaN when too few samples."""
    s = SITES[SITE]
    if SITE == "dkasc38":
        clean = pd.read_parquet(CLEAN).loc[s["span"][0]:]
        p = clean.power.where(clean.valid)
        return p.resample("h").mean().where(p.resample("h").count() >= 10)
    if "elb" in s:
        return elb_hourly(s) * CAP / (kwp or s["kwp"] or 1.0)
    h = pvdaq_hourly(s)
    return (h.p_w.where(h.n_min >= 50) / 1000 * CAP / (kwp or s["kwp"])).clip(lower=0)  # night standby draw (-3 W) -> 0


def build(site: str = "dkasc38", kwp: float = None):
    use_site(site)
    target = target_hourly(kwp)
    cs5 = clearsky_5min(*SITES[site]["span"])
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
    for name, (a, b) in (SPLIT if site == "dkasc38" else SITES[site].get("split", US_SPLIT)).items():
        df.loc[a:b, "split"] = name
    df = df[df.split.notna()]
    df.index.name = "time"
    return df


if __name__ == "__main__":
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else "dkasc38"
    cal = "--calibrate" in sys.argv  # US only: effective capacity from pre-test history instead of nameplate
    assert cal or SITES[site]["kwp"], f"{site}: nameplate unknown, run with --calibrate"
    kwp = effective_kwp(site) if cal else None
    if cal:
        print(f"{site}: nameplate {SITES[site]['kwp']} kWp -> effective {kwp:.2f} kWp (pre-test calibration data)")
    df = build(site, kwp)
    if site != "dkasc38":
        df = df[df.split == "test"]  # calibration rows stay internal (history_rows); downstream reads test rows only
    df.attrs["kwp"] = kwp or SITES[site]["kwp"]
    day = df[df.zenith < 85]
    print(df.shape, "days:", df.day.nunique())
    print(day.groupby("split").agg(rows=("y", "size"), y_ok=("y", lambda s: s.notna().mean()),
                                   gfs_ok=("ghi_gfs", lambda s: s.notna().mean())).round(3))
    r = day[["y", "p_phys1", "p_lag"]].dropna()
    print("corr(y, gfs physics) %.3f  corr(y, yesterday) %.3f" % (r.y.corr(r.p_phys1), r.y.corr(r.p_lag)))
    # self-checks: no lookahead in the lag, clear-sky >= GFS on average, physics in the right scale
    t = df.index[df.hour.eq(18)][5]
    assert df.loc[t, "p_lag"] == df.y.get(t - pd.Timedelta(days=2)) or np.isnan(df.loc[t, "p_lag"])
    assert (day.poa_cs.mean() >= day.poa_gfs.mean()) and 0.6 < (r.y.sum() / r.p_phys1.sum()) < 1.2  # unit-error guard (ELB 1.04: GFS dimmer than Michigan sky)
    if site == "dkasc38":
        assert df.loc["2023-01-01":, "split"].eq("test").all() and not df.loc[:"2022-12-31", "split"].eq("test").any()
    dk = day.y.groupby(day.day).sum() / day.p_cs1.groupby(day.day).sum()
    c = day[day.day.isin(dk[dk > min(0.88, dk.quantile(0.9))].index) & day.hour.between(8, 16)]  # site-relative for weak arrays
    kh = c.k.groupby(c.hour).median()
    print("clear-day median GFS k by hour:", kh.round(2).to_dict())
    assert kh.max() - kh.min() < 0.2, "GFS clearness drifts through the day: timing misalignment"
    out = SITES[site]["out"].replace(".parquet", "_cal.parquet") if cal else SITES[site]["out"]
    df.to_parquet(out)
    print("saved ->", out)
