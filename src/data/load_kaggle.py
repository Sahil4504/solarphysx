from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

# Formats differ per file. If parsing throws an error, open the CSV and check.
FORMATS = {
    ("1", "Generation"): "%d-%m-%Y %H:%M",
    ("1", "Weather_Sensor"): "%Y-%m-%d %H:%M:%S",
    ("2", "Generation"): "%Y-%m-%d %H:%M:%S",
    ("2", "Weather_Sensor"): "%Y-%m-%d %H:%M:%S",
}


def load_plant(plant: str) -> pd.DataFrame:
    gen = pd.read_csv(RAW / f"Plant_{plant}_Generation_Data.csv")
    wx = pd.read_csv(RAW / f"Plant_{plant}_Weather_Sensor_Data.csv")
    gen["DATE_TIME"] = pd.to_datetime(
        gen["DATE_TIME"], format=FORMATS[(plant, "Generation")]
    )
    wx["DATE_TIME"] = pd.to_datetime(
        wx["DATE_TIME"], format=FORMATS[(plant, "Weather_Sensor")]
    )
    wx = wx.drop(columns=["SOURCE_KEY"])  # one weather sensor per plant
    df = gen.merge(wx, on=["DATE_TIME", "PLANT_ID"], how="left")
    df["plant"] = int(plant)
    return df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)


def sanity_report(df: pd.DataFrame) -> None:
    p = df["plant"].iloc[0]
    print(f"\n=== Plant {p} ===")
    print("rows:", len(df), "| inverters:", df["SOURCE_KEY"].nunique())
    print("span:", df["DATE_TIME"].min(), "->", df["DATE_TIME"].max())

    day = df[df["AC_POWER"] > 0]
    print("median DC/AC ratio:", round((day["DC_POWER"] / day["AC_POWER"]).median(), 3))

    full = pd.date_range(df["DATE_TIME"].min(), df["DATE_TIME"].max(), freq="15min")
    print("expected steps per inverter:", len(full))
    print(
        df.groupby("SOURCE_KEY")["DATE_TIME"]
        .nunique()
        .describe()[["min", "mean", "max"]]
    )

    print("rows missing weather:", df["IRRADIATION"].isna().sum())
    print("max irradiation:", df["IRRADIATION"].max())
    print("negative DC rows:", (df["DC_POWER"] < 0).sum())


if __name__ == "__main__":
    for p in ["1", "2"]:
        d = load_plant(p)
        sanity_report(d)
        d.to_parquet(f"data/interim/plant{p}.parquet")
