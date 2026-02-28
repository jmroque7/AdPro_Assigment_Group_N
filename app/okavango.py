from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd

def download_all_datasets(downloads_dir: Path, urls: dict[str, str]) -> None:
   
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        path = downloads_dir / filename
        if path.exists() and path.stat().st_size > 0:
            continue  # avoid redownload

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)


OWID_URLS = {
    "annual-change-forest-area.csv": "https://ourworldindata.org/grapher/annual-change-forest-area.csv",
    "annual-deforestation.csv": "https://ourworldindata.org/grapher/annual-deforestation.csv",
    "terrestrial-protected-areas.csv": "https://ourworldindata.org/grapher/terrestrial-protected-areas.csv",
    "share-degraded-land.csv": "https://ourworldindata.org/grapher/share-degraded-land.csv",
    "forest-area-as-share-of-land-area.csv": "https://ourworldindata.org/grapher/forest-area-as-share-of-land-area.csv",
}

def merge_world_with_dataset(world, df_latest):
    world = world.copy()
    world = world.rename(columns={"ISO_A3": "iso3"})
    merged = world.merge(df_latest, on="iso3", how="left")
    return merged