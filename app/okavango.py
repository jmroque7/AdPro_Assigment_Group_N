"""
Okavango data utilities.

This module provides:
- Function 1: download_all_datasets
- Function 2: merge_world_with_dataset
- Class: OkavangoProject (executes Function 1 and Function 2 in __init__)

PEP8 compliant names, type hints included, and a small Pydantic config for validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests
from pydantic import BaseModel, Field
import matplotlib.pyplot as pltimport re
from difflib import get_close_matches

# Manual aliases for the most common name mismatches
NAME_ALIASES: dict[str, str] = {
    # Natural Earth -> OWID style (or vice-versa). Normalize both sides anyway.
    "united states of america": "united states",
    "russian federation": "russia",
    "viet nam": "vietnam",
    "czechia": "czech republic",
    "bolivia plurinational state of": "bolivia",
    "iran islamic republic of": "iran",
    "venezuela bolivarian republic of": "venezuela",
    "tanzania united republic of": "tanzania",
    "korea republic of": "south korea",
    "korea democratic people's republic of": "north korea",
}

# Things we should never map as "countries" in a fallback
REGION_KEYWORDS = {
    "africa", "europe", "asia", "north america", "south america", "oceania",
    "world", "international", "high income", "low income", "upper middle income", "lower middle income",
    "european union",
}

def normalize_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s-]", "", x)      # remove punctuation
    x = re.sub(r"\s+", " ", x)         # collapse spaces
    x = NAME_ALIASES.get(x, x)         # apply alias
    return x

def is_region_like(name: str) -> bool:
    n = normalize_name(name)
    return n in REGION_KEYWORDS


OWID_URLS: dict[str, str] = {
    "annual-change-forest-area.csv": (
        "https://ourworldindata.org/grapher/annual-change-forest-area.csv"
    ),
    "annual-deforestation.csv": "https://ourworldindata.org/grapher/annual-deforestation.csv",
    "terrestrial-protected-areas.csv": (
        "https://ourworldindata.org/grapher/terrestrial-protected-areas.csv"
    ),
    "share-degraded-land.csv": "https://ourworldindata.org/grapher/share-degraded-land.csv",
    "forest-area-as-share-of-land-area.csv": (
        "https://ourworldindata.org/grapher/forest-area-as-share-of-land-area.csv"
    ),
    "ne_110m_admin_0_countries.zip": (
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    ),
}


# ---------------------------
# Function 1
# ---------------------------
def download_all_datasets(downloads_dir: Path, urls: dict[str, str]) -> None:
    """
    Download all datasets in `urls` into `downloads_dir` (skips files already present).
    """
    downloads_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        path = downloads_dir / filename
        if path.exists() and path.stat().st_size > 0:
            continue  # avoid redownload

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)


# ---------------------------
# Function 2 helpers
# ---------------------------
def get_latest_year_slice(df: pd.DataFrame, year_col: str = "Year") -> pd.DataFrame:
    """
    Return only rows belonging to the latest year in the dataframe (OWID datasets).
    """
    out = df.copy()
    if year_col not in out.columns:
        return out

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    latest_year = out[year_col].max()
    return out[out[year_col] == latest_year]


def pick_world_iso_col(world: gpd.GeoDataFrame) -> str:
    """
    Return the best ISO3 column name for the Natural Earth world dataset.
    """
    candidates = ["ISO_A3", "ADM0_A3", "ISO_A3_EH", "ADM0_A3_US"]
    for col in candidates:
        if col in world.columns:
            return col
    raise ValueError(
        "No ISO3 column found in world GeoDataFrame. "
        "Expected one of: ISO_A3, ADM0_A3, ISO_A3_EH, ADM0_A3_US"
    )


def detect_value_column(df: pd.DataFrame) -> str:
    drop = {"Entity", "Code", "iso3", "Year"}
    candidates = [c for c in df.columns if c not in drop]
    if not candidates:
        raise ValueError("No metric column found in dataset.")
    # OWID grapher metric is almost always the last non-meta column
    return candidates[-1]


def merge_world_with_dataset(
    world: gpd.GeoDataFrame,
    dataset_df: pd.DataFrame,
    *,
    owid_iso_col: str = "Code",
    latest_year_only: bool = True,
) -> gpd.GeoDataFrame:
    """
    Merge a world GeoDataFrame with an OWID grapher dataset.

    Requirements satisfied:
    - Uses geopandas
    - Left dataframe is the GeoDataFrame (world)
    - Ensures join key aligns (ISO3 codes)
    """
    if not isinstance(world, gpd.GeoDataFrame):
        raise TypeError("world must be a GeoDataFrame")

    world_gdf = world.copy()
    world_iso_col = pick_world_iso_col(world_gdf)
    world_gdf["iso3"] = world_gdf[world_iso_col].astype(str)

    df = dataset_df.copy()

    if latest_year_only:
        df = get_latest_year_slice(df, "Year")

    # Normalize OWID ISO column to "iso3"
    if owid_iso_col != "iso3":
        df = df.rename(columns={owid_iso_col: "iso3"})

    if "iso3" not in df.columns:
        raise ValueError("OWID dataset missing ISO3 column (expected 'Code').")

    df["iso3"] = df["iso3"].astype(str)
    df = df[(df["iso3"] != "nan") & (df["iso3"].str.len() == 3)]

    value_col = detect_value_column(df)

    # Keep only useful columns and avoid duplicates
    keep_cols = ["iso3"]
    for col in ("Entity", "Year", value_col):
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].drop_duplicates(subset=["iso3"])

    merged = world_gdf.merge(df, on="iso3", how="left")
    return merged


# ---------------------------
# Pydantic config + Class
# ---------------------------
class OkavangoConfig(BaseModel):
    """
    Configuration for OkavangoProject.

    You can pass paths as strings; Pydantic will validate/convert.
    """

    project_root: Path = Field(default=Path("."))
    downloads_dirname: str = Field(default="downloads")
    natural_earth_zip_name: str = Field(default="ne_110m_admin_0_countries.zip")
    latest_year_only: bool = Field(default=True)
    download: bool = Field(default=True)


class OkavangoProject:
    """
    Project handler that downloads, loads, and merges datasets.

    Assignment requirements met:
    - Class exists to handle the data
    - __init__ executes Function 1 and Function 2
    - __init__ reads datasets into dataframes as attributes
    """

    def __init__(self, config: OkavangoConfig | None = None) -> None:
        self.config = config or OkavangoConfig()
        self.project_root: Path = self.config.project_root.resolve()
        self.downloads_dir: Path = self.project_root / self.config.downloads_dirname

        # Execute Function 1 in __init__
        if self.config.download:
            download_all_datasets(self.downloads_dir, OWID_URLS)

        # Unzip Natural Earth if needed
        self._extract_natural_earth_zip()

        # Read datasets into attributes
        self.datasets: Dict[str, pd.DataFrame] = self._load_owid_datasets()

        # Load world map GeoDataFrame
        self.world: gpd.GeoDataFrame = self._load_world_map()

        # Execute Function 2 in __init__ (GeoPandas LEFT)
        self.merged_maps: Dict[str, gpd.GeoDataFrame] = {
            name: merge_world_with_dataset(
                self.world,
                df,
                latest_year_only=self.config.latest_year_only,
            )
            for name, df in self.datasets.items()
        }

    def _extract_natural_earth_zip(self) -> None:
        zip_path = self.downloads_dir / self.config.natural_earth_zip_name
        if not zip_path.exists():
            return

        shp_path = self.downloads_dir / "ne_110m_admin_0_countries.shp"
        if shp_path.exists():
            return

        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(self.downloads_dir)

    def _load_owid_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load only the CSV datasets from OWID_URLS into a dict of DataFrames.
        """
        out: Dict[str, pd.DataFrame] = {}
        for filename in OWID_URLS.keys():
            if not filename.endswith(".csv"):
                continue
            path = self.downloads_dir / filename
            out[filename] = pd.read_csv(path)
        return out

    def _load_world_map(self) -> gpd.GeoDataFrame:
        """
        Load the Natural Earth world map from the extracted shapefile.
        """
        shp_path = self.downloads_dir / "ne_110m_admin_0_countries.shp"
        if not shp_path.exists():
            raise FileNotFoundError(
                "World shapefile not found. Expected: downloads/ne_110m_admin_0_countries.shp"
            )
        return gpd.read_file(shp_path)


if __name__ == "__main__":
    project = OkavangoProject()

    print("Datasets loaded:")
    print(project.datasets.keys())

    print("\nMerge check (non-null values per map):")
    for name, gdf in project.merged_maps.items():
        metric_col = detect_value_column(project.datasets[name])
        non_null = int(gdf[metric_col].notna().sum()) if metric_col in gdf.columns else 0
        print(f"{name}: {non_null} matched countries (metric: {metric_col})")

    # Optional: show columns for the first merged GeoDataFrame
    first_name = next(iter(project.merged_maps.keys()))
    print("\nMerged columns sample:", list(project.merged_maps[first_name].columns))

if __name__ == "__main__":
    project = OkavangoProject()

    # Pick one dataset to visualize
    dataset_name = "annual-deforestation.csv"
    gdf = project.merged_maps[dataset_name]

    # Detect correct metric column
    metric_col = detect_value_column(project.datasets[dataset_name])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    gdf.plot(
        column=metric_col,
        cmap="viridis",
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "No data",
        },
    )

    ax.set_title(f"{dataset_name} (Latest Year)", fontsize=14)
    ax.set_axis_off()

    plt.show()
