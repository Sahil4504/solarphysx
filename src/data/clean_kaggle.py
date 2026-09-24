from pathlib import Path

import pandas as pd

from src.data.load_kaggle import RAW, FORMATS

OUT = Path("data/interim")
START, END = "2020-05-15 00:00", "2020-06-17 23:45"
FREQ = "15min"
MAX_INTERP_STEPS = 2  # fill weather gaps up to 30 min only
DC_SCALE = {"1": 0.1, "2": 1.0}  # Plant 1 DC is 10x inflated (EDA: DC/AC slope = 10)
WX_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# Fault rules (EDA: DC collapses while the sun is strong and siblings keep producing)
SUN_MIN = 0.2  # kW/m2, the sun is clearly up
ZERO_DC = 1.0  # kW, effectively zero output
SIBLING_MIN = 50.0  # kW, siblings are clearly producing
SIBLING_FRAC = 0.2  # below 20% of the sibling median counts as a fault


def full_grid() -> pd.DatetimeIndex:
    return pd.date_range(START, END, freq=FREQ, name="DATE_TIME")


def clean_weather(plant: str, grid: pd.DatetimeIndex) -> pd.DataFrame:
    wx = pd.read_csv(RAW / f"Plant_{plant}_Weather_Sensor_Data.csv")
    wx["DATE_TIME"] = pd.to_datetime(
        wx["DATE_TIME"], format=FORMATS[(plant, "Weather_Sensor")]
    )
    wx = wx.drop_duplicates("DATE_TIME").set_index("DATE_TIME").reindex(grid)
    wx["wx_observed"] = wx["IRRADIATION"].notna()
    wx[WX_COLS] = wx[WX_COLS].interpolate(
        method="time", limit=MAX_INTERP_STEPS, limit_area="inside"
    )
    return wx[WX_COLS + ["wx_observed"]].reset_index()


def clean_generation(plant: str, grid: pd.DatetimeIndex) -> pd.DataFrame:
    gen = pd.read_csv(RAW / f"Plant_{plant}_Generation_Data.csv")
    gen["DATE_TIME"] = pd.to_datetime(
        gen["DATE_TIME"], format=FORMATS[(plant, "Generation")]
    )
    gen["DC_POWER"] = gen["DC_POWER"] * DC_SCALE[plant]
    gen = gen.drop_duplicates(["SOURCE_KEY", "DATE_TIME"])

    # Last use of AC: confirm the DC correction, then drop it (AC is derived from DC)
    day = gen[gen["AC_POWER"] > 0]
    ratio = (day["DC_POWER"] / day["AC_POWER"]).median()
    print(f"Plant {plant}: median DC/AC after scaling = {ratio:.3f}")
    assert 0.95 < ratio < 1.10, f"Plant {plant}: DC/AC = {ratio:.3f}, scaling is wrong"

    keys = sorted(gen["SOURCE_KEY"].unique())
    idx = pd.MultiIndex.from_product([keys, grid], names=["SOURCE_KEY", "DATE_TIME"])
    gen = gen.set_index(["SOURCE_KEY", "DATE_TIME"])[["DC_POWER"]].reindex(idx)
    gen["gen_observed"] = gen["DC_POWER"].notna()
    return gen.reset_index()


def add_fault_flag(df: pd.DataFrame) -> pd.DataFrame:
    sib = df.groupby("DATE_TIME")["DC_POWER"].transform("median")
    sunny = df["IRRADIATION"] > SUN_MIN
    dead = sunny & (df["DC_POWER"] < ZERO_DC)
    lagging = (sib > SIBLING_MIN) & (df["DC_POWER"] < SIBLING_FRAC * sib)
    df["sibling_median_dc"] = sib
    df["fault"] = df["gen_observed"] & (dead | lagging)
    return df


def clean_plant(plant: str) -> pd.DataFrame:
    grid = full_grid()
    gen = clean_generation(plant, grid)
    wx = clean_weather(plant, grid)
    df = gen.merge(wx, on="DATE_TIME", how="left")
    df = add_fault_flag(df)
    df["plant"] = int(plant)
    df["valid"] = df["gen_observed"] & df["IRRADIATION"].notna() & ~df["fault"]
    return df


def longest_run(mask: pd.Series) -> int:
    runs = (mask != mask.shift()).cumsum()
    return int(mask.groupby(runs).sum().max())


def report(df: pd.DataFrame) -> None:
    p = df["plant"].iloc[0]
    print(f"\n=== Plant {p} (cleaned) ===")
    print("grid rows:", len(df))
    print(
        "observed:",
        int(df["gen_observed"].sum()),
        "| fault:",
        int(df["fault"].sum()),
        "| valid:",
        int(df["valid"].sum()),
    )

    sunny_obs = df["gen_observed"] & (df["IRRADIATION"] > SUN_MIN)
    print(
        f"fault share of sunny observed rows: {100 * df.loc[sunny_obs, 'fault'].mean():.1f}%"
    )

    wx = df.drop_duplicates("DATE_TIME")
    print("weather steps never observed:", int((~wx["wx_observed"]).sum()))
    print(
        "weather steps still missing after interp:", int(wx["IRRADIATION"].isna().sum())
    )

    g = df.groupby("SOURCE_KEY")
    per_inv = pd.DataFrame(
        {
            "missing_pct": 100 * (1 - g["gen_observed"].mean()),
            "fault_pct": 100 * g["fault"].sum() / g["gen_observed"].sum(),
            "longest_gap_steps": g["gen_observed"].apply(lambda s: longest_run(~s)),
        }
    ).round(1)
    print("\nworst 5 by missing:")
    print(per_inv.sort_values("missing_pct", ascending=False).head(5))
    print("\nworst 5 by fault:")
    print(per_inv.sort_values("fault_pct", ascending=False).head(5))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in ["1", "2"]:
        d = clean_plant(p)
        report(d)
        d.to_parquet(OUT / f"clean_plant{p}.parquet", index=False)
