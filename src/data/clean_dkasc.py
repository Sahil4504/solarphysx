"""Clean DKASC array #38 (Q CELLS 5.9 kW mono-Si, fixed): decisions from notebooks/02_eda_dkasc.ipynb.

1. Full 5-min grid 2018-01-01 -> 2023-09-30, local time ACST (timestamps checked against GFS/UTC).
2. Temperatures outside [-5, 50] C masked; gaps <= 30 min interpolated.
3. Panel tilt/azimuth fitted from periods when the tilted (POA) sensor works.
4. POA rebuilt from horizontal sensors (GHI, DHI) with pvlib where the POA sensor was down.
5. Flags: poa_rebuilt, fault (sun clearly up, array ~off), valid (everything usable).
Output: data/interim/dkasc38_clean.parquet.   Run + self-check:  python -m src.data.clean_dkasc
"""

import numpy as np
import pandas as pd
import pvlib

RAW = "data/raw/dkasc/59-Site_DKA-M19_C-Phase.csv"
OUT = "data/interim/dkasc38_clean.parquet"
LAT, LON, TZ = -23.762, 133.875, "Australia/Darwin"  # ACST, UTC+9:30, no daylight saving
CAP = 5.9  # kW nameplate
START, END = "2018-01-01", "2023-09-30 23:55"
COLS = {"Active_Power": "power", "Global_Horizontal_Radiation": "ghi", "Diffuse_Horizontal_Radiation": "dhi",
        "Radiation_Global_Tilted": "poa_meas", "Weather_Temperature_Celsius": "temp",
        "Weather_Relative_Humidity": "rh", "Wind_Speed": "wind"}


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW, on_bad_lines="skip", low_memory=False, usecols=["timestamp", *COLS], parse_dates=["timestamp"],
                     index_col="timestamp").rename(columns=COLS).sort_index()
    df = df[~df.index.duplicated()].apply(pd.to_numeric, errors="coerce")
    return df.reindex(pd.date_range(START, END, freq="5min", name="time"))  # gaps become NaN rows


def sun(index: pd.DatetimeIndex) -> pd.DataFrame:
    sp = pvlib.solarposition.get_solarposition(index.tz_localize(TZ), LAT, LON)
    return pd.DataFrame({"zenith": sp.apparent_zenith.to_numpy(), "sun_az": sp.azimuth.to_numpy()}, index=index)


def poa_model(df: pd.DataFrame, tilt: float, az: float) -> pd.Series:
    cz = np.cos(np.radians(df.zenith)).clip(lower=0.087)  # cap at 85 deg to avoid blow-up near the horizon
    dni = ((df.ghi - df.dhi) / cz).clip(lower=0)
    extra = pvlib.irradiance.get_extra_radiation(df.index.dayofyear)
    tot = pvlib.irradiance.get_total_irradiance(tilt, az, df.zenith, df.sun_az, dni, df.ghi, df.dhi,
                                                dni_extra=extra, model="haydavies")
    return tot["poa_global"].where(df.zenith < 90, 0.0).fillna(0.0)


def fit_orientation(df: pd.DataFrame) -> tuple[float, float, float]:
    ok = (df.zenith < 75) & (df.ghi > 50) & (df.dhi > 0) & (df.dhi < df.ghi) & (df.poa_meas > 20)
    s = df[ok].loc["2021":"2022"].sample(20000, random_state=0)
    def rmse(t, a): return float(np.sqrt(((poa_model(s, t, a) - s.poa_meas) ** 2).mean()))
    t0, a0 = min(((t, a) for t in range(0, 46, 2) for a in range(300, 421, 10)), key=lambda p: rmse(*p))
    t, a = min(((t, a) for t in np.arange(t0 - 2, t0 + 2.1, 0.5) for a in range(a0 - 10, a0 + 11, 2)), key=lambda p: rmse(*p))
    return float(t), float(a % 360), rmse(t, a)


def clean() -> tuple[pd.DataFrame, dict]:
    df = load().join(sun(pd.date_range(START, END, freq="5min", name="time")))
    rep = {}
    bad_t = (df.temp < -5) | (df.temp > 50)
    rep["temp_masked"] = int(bad_t.sum())
    df.loc[bad_t, "temp"] = np.nan
    df["temp"] = df.temp.interpolate(method="time", limit=6, limit_area="inside")
    df["power"] = df.power.clip(lower=0)

    tilt, az, fit_rmse = fit_orientation(df)
    rep.update(tilt=tilt, azimuth=az, fit_rmse_Wm2=round(fit_rmse, 1))
    df["poa_model"] = poa_model(df, tilt, az)
    down = df.poa_meas.isna() | ((df.poa_meas < 10) & (df.ghi > 200))  # sensor missing or reading ~0 in full sun
    df["poa_rebuilt"] = down & df.ghi.notna() & df.dhi.notna()
    df["poa"] = df.poa_meas.where(~down, df.poa_model.where(df.poa_rebuilt))
    rep["poa_rebuilt_daytime_rows"] = int((df.poa_rebuilt & (df.zenith < 85)).sum())

    df["fault"] = df.power.notna() & (df.poa > 300) & (df.power < 0.05 * CAP)
    df["valid"] = df.power.notna() & df.poa.notna() & df.temp.notna() & ~df.fault
    rep["fault_rows"] = int(df.fault.sum())
    rep["valid_daytime_share_by_year"] = (df[df.zenith < 85].groupby(df[df.zenith < 85].index.year).valid.mean()
                                          .round(3).to_dict())
    return df[["power", "poa", "poa_meas", "poa_model", "ghi", "dhi", "temp", "rh", "wind", "zenith", "sun_az",
               "poa_rebuilt", "fault", "valid"]], rep


if __name__ == "__main__":
    df, rep = clean()
    for k, v in rep.items():
        print(f"{k}: {v}")
    # self-check: plausible orientation for a fixed array at 23.8 S (faces north), rebuild accurate where both exist
    assert 5 <= rep["tilt"] <= 40 and (rep["azimuth"] <= 45 or rep["azimuth"] >= 315), rep
    both = (df.zenith < 75) & df.poa_meas.gt(20) & ~df.poa_rebuilt & df.ghi.gt(50)
    err = (df.poa_model - df.poa_meas)[both].loc["2023"]  # year not used in the fit
    print(f"rebuild check on 2023 (not used in fit): bias {err.mean():.1f} W/m2, rmse {np.sqrt((err**2).mean()):.1f} W/m2")
    assert abs(err.mean()) < 25 and np.sqrt((err ** 2).mean()) < 80
    assert df.index.is_monotonic_increasing and (df.index.to_series().diff().dropna() == pd.Timedelta("5min")).all()
    df.to_parquet(OUT)
    print("saved ->", OUT, df.shape)
