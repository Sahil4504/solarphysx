"""Download NREL PVDAQ daily CSVs + system metadata (OEDI, CC BY 4.0) for the US transfer test. Resumable.

Default: 1423 = RTC-NV-Baseline, Henderson NV (6 kW, tilt 35, az 180); 50 = NREL x-Si 6, Golden CO (6 kW, tilt 45, az 158);
years 2021-2023 (GFS archive on AWS starts 2021). Files land in data/raw/pvdaq/<system_id>/.
  python -m src.data.fetch_pvdaq                     python -m src.data.fetch_pvdaq --systems 1423 --years 2022 2023
"""

import argparse
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://oedi-data-lake.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OUT = Path("data/raw/pvdaq")


def _open(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:  # dropped connections happen on long runs; back off and retry
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} after {e}", flush=True)
            time.sleep(2 ** i)


def parse_listing(xml: bytes):
    """(key, size) pairs and the continuation token (None when the listing is complete)."""
    root = ET.fromstring(xml)
    items = [(c.findtext(NS + "Key"), int(c.findtext(NS + "Size"))) for c in root.iter(NS + "Contents")]
    token = root.findtext(NS + "NextContinuationToken") if root.findtext(NS + "IsTruncated") == "true" else None
    return items, token


def keys(prefix):
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
        items, token = parse_listing(_open(f"{BASE}/?{urllib.parse.urlencode(q)}"))
        yield from items
        if not token:
            return


def get(key, dest: Path, size=None) -> bool:
    """Download unless an identical-size copy exists; writes via .part so an interrupted file is never kept."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(_open(f"{BASE}/{urllib.parse.quote(key)}"))
    tmp.replace(dest)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=int, nargs="*", default=[1423, 50])
    ap.add_argument("--years", type=int, nargs="*", default=[2021, 2022, 2023])
    a = ap.parse_args()
    get("pvdaq/csv/systems_20250729.csv", OUT / "systems_20250729.csv")
    for sid in a.systems:
        get(f"pvdaq/csv/system_metadata/{sid}_system_metadata.json", OUT / str(sid) / f"{sid}_system_metadata.json")
        for y in a.years:
            got = n = 0
            for key, size in keys(f"pvdaq/csv/pvdata/system_id={sid}/year={y}/"):
                n += 1
                got += get(key, OUT / str(sid) / Path(key).name, size)
            print(f"system {sid} {y}: {n} daily files ({got} new)", flush=True)
