"""Download one month of NYC TLC green-taxi trips + zone lookup into data/raw (git-ignored)."""
import pathlib, urllib.request
BASE = "https://d37ci6vzurychx.cloudfront.net"
FILES = {"green_tripdata_2024-01.parquet": f"{BASE}/trip-data/green_tripdata_2024-01.parquet",
         "taxi_zone_lookup.csv": f"{BASE}/misc/taxi_zone_lookup.csv"}
raw = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dest = raw / name
    if not dest.exists():
        print("downloading", url); urllib.request.urlretrieve(url, dest)
    print(dest, dest.stat().st_size, "bytes")
