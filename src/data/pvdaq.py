"""NREL PVDAQ systems for the US zero-shot transfer test: minute loader + hourly / daily aggregates for EDA.

Raw: data/raw/pvdaq/<id>/system_<id>__date_YYYY_MM_DD.csv (1-min, from src.data.fetch_pvdaq). No cleaning here:
values are aggregated as recorded (W, W/m2, C) so the EDA notebook sees the raw behaviour.
  python -m src.data.pvdaq        -> data/interim/pvdaq_<id>_hourly.csv, pvdaq_<id>_daily.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW, INTERIM = Path("data/raw/pvdaq"), Path("data/interim")
# metadata from <id>_system_metadata.json; channel names from the CSV headers
SITES = {
    1423: dict(name="Henderson NV (RTC-NV-Baseline)", lat=36.0275, lon=-114.9215, alt=697.0, tilt=35.0, az=180.0, kwp=6.0,
               power=["inv1_ac_power__4854", "inv2_ac_power__4860"], poa="poa_irradiance__4861",
               t_amb="ambient_temp__4864", t_mod="module_temp_1__4865"),
    50: dict(name="Golden CO (NREL x-Si 6)", lat=39.742, lon=-105.1727, alt=1994.7, tilt=45.0, az=158.0, kwp=6.0,
             power=["ac_power__752"], poa="poa_irradiance__750", t_amb="ambient_temp__759", t_mod="module_temp_1__760"),
}


def load_minutes(sid: int) -> pd.DataFrame:
    """All daily files of one system as 1-min rows: p_w (AC, summed over inverters), poa, t_amb, t_mod."""
    s = SITES[sid]
    cols = ["measured_on", *s["power"], s["poa"], s["t_amb"], s["t_mod"]]
    # some daily files lack channels (Golden: 41 of 524); missing ones become NaN instead of failing
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols).reindex(columns=cols)
                    for f in sorted((RAW / str(sid)).glob("system_*.csv"))])
    df.index = pd.to_datetime(df.pop("measured_on"))
    out = pd.DataFrame({"p_w": df[s["power"]].sum(axis=1, min_count=len(s["power"])),
                        "poa": df[s["poa"]], "t_amb": df[s["t_amb"]], "t_mod": df[s["t_mod"]]})
    return out[~out.index.duplicated()].sort_index()


def aggregates(m: pd.DataFrame):
    """Hourly means (+ minutes present) and per-day energy / power-weighted centre time (timezone check)."""
    h = m.resample("h").mean().join(m.p_w.resample("h").count().rename("n_min"))
    pos = m.p_w.clip(lower=0)
    minute = m.index.hour * 60 + m.index.minute
    d = pd.DataFrame({"kwh": pos.resample("D").sum() / 60 / 1000, "poa_kwh_m2": m.poa.clip(lower=0).resample("D").sum() / 60 / 1000,
                      "n_min": m.p_w.resample("D").count(),
                      "centroid_min": (pos * minute).resample("D").sum() / pos.resample("D").sum().replace(0, np.nan)})
    return h, d


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    for sid in SITES:
        m = load_minutes(sid)
        h, d = aggregates(m)
        h.to_csv(INTERIM / f"pvdaq_{sid}_hourly.csv"); d.to_csv(INTERIM / f"pvdaq_{sid}_daily.csv")
        assert h.n_min.max() <= 60 and len(d) >= d.n_min.gt(0).sum()
        print(f"{sid}: {len(m):,} minutes, {d.n_min.gt(0).sum()} days with data, {m.index.min()} .. {m.index.max()}, "
              f"peak AC {m.p_w.max():.0f} W, min POA {m.poa.min():.0f}")
