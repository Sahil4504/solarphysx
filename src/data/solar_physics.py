"""Solar-geometry features for the Kaggle plants (pvlib).

The dataset gives no coordinates, so each plant's location is fitted from its own data:
on the 5 smoothest (clearest) days, choose the lat/lon whose clear-sky curve best matches
the measured irradiation. Timestamps are treated as India Standard Time (UTC+05:30).

Because the timestamp convention (interval start vs end) is unknown, the fitted longitude
is an EFFECTIVE longitude: it absorbs any fixed timing offset, which keeps the solar features
aligned with the data (what the model needs), not necessarily the true map position.

Run to (re)fit and self-check:
    python -m src.data.solar_physics
"""

import numpy as np
import pandas as pd
import pvlib

from src.data.load_kaggle import FORMATS, RAW

TZ = "Asia/Kolkata"
# Fitted by fit_location() on 2026-09-22 (see vault: Datasets). Re-run this module to refit.
LOCATIONS = {1: (11.5, 79.3), 2: (12.0, 80.3)}  # (lat, effective lon)
CSI_MIN_GHI = 0.05  # kW/m2; below this the clear-sky index is meaningless (sun near horizon)


def clear_sky(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """cos(zenith), azimuth and clear-sky GHI (kW/m2) at naive local timestamps."""
    t = times.tz_localize(TZ)
    sp = pvlib.solarposition.get_solarposition(t, lat, lon)
    # ponytail: Haurwitz clear-sky needs zenith only; swap to Ineichen if turbidity data is added
    ghi = pvlib.clearsky.haurwitz(sp["apparent_zenith"])["ghi"] / 1000
    return pd.DataFrame({
        "cos_zenith": np.cos(np.radians(sp["apparent_zenith"])).clip(lower=0).to_numpy(),
        "azimuth": sp["azimuth"].to_numpy(),
        "ghi_cs": ghi.to_numpy(),
    }, index=times)


def clearest_days(s: pd.Series, n: int = 5) -> pd.Series:
    by_day = s.groupby(s.index.date)
    rough = by_day.apply(lambda x: np.abs(np.diff(x.to_numpy(), 2)).sum() / max(x.sum(), 1e-6))
    full = by_day.size() >= 90
    days = rough[full].nsmallest(n).index
    return s[np.isin(s.index.date, days)]


def fit_location(plant: int) -> tuple[float, float]:
    wx = pd.read_csv(RAW / f"Plant_{plant}_Weather_Sensor_Data.csv")
    wx["DATE_TIME"] = pd.to_datetime(wx["DATE_TIME"], format=FORMATS[(str(plant), "Weather_Sensor")])
    s = clearest_days(wx.drop_duplicates("DATE_TIME").set_index("DATE_TIME")["IRRADIATION"])

    def score(lat, lon):
        return np.corrcoef(s.to_numpy(), clear_sky(s.index, lat, lon)["ghi_cs"].to_numpy())[0, 1]

    def daylight_gap(lat, lon):
        model = (clear_sky(s.index, lat, lon)["ghi_cs"] > 0.02).sum()
        return abs(model - (s > 0.02).sum())

    # Longitude sets timing: strongly identified by curve correlation.
    # Latitude barely changes the curve shape (r moves < 0.01 over 0-28 deg), so it is fitted
    # from day length instead. ponytail: 15-min resolution makes it approximate (+/- ~5 deg).
    lon = max(np.arange(70, 95.01, 0.25), key=lambda x: score(15, x))
    lon = max(np.arange(lon - 1, lon + 1.01, 0.05), key=lambda x: score(15, x))
    lat = min(np.arange(5, 35.01, 0.5), key=lambda y: daylight_gap(y, lon))
    print(f"plant {plant}: lat {lat:.1f}, effective lon {lon:.2f}, r = {score(lat, lon):.4f}")
    return round(float(lat), 1), round(float(lon), 2)


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cos_zenith, azimuth, ghi_cs and clear-sky index (csi) for each row's plant."""
    parts = []
    for plant, g in df.groupby("plant"):
        times = pd.DatetimeIndex(g["DATE_TIME"].unique()).sort_values()
        cs = clear_sky(times, *LOCATIONS[int(plant)])
        parts.append(g.join(cs, on="DATE_TIME"))
    out = pd.concat(parts).loc[df.index]
    out["csi"] = np.where(out["ghi_cs"] > CSI_MIN_GHI,
                          out["IRRADIATION"] / out["ghi_cs"].clip(lower=CSI_MIN_GHI), 0.0)
    out["csi"] = out["csi"].clip(0, 1.5)  # ponytail: clip hides sensor spikes; revisit if ramps need >1.5
    return out


if __name__ == "__main__":
    fitted = {p: fit_location(p) for p in (1, 2)}
    print("LOCATIONS =", fitted)
    for p, (lat, lon) in fitted.items():
        assert 6 < lat < 36 and 68 < lon < 98, f"plant {p}: fit outside India, check timestamps"
