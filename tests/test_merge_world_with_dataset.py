from __future__ import annotations
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from app.okavango import merge_world_with_dataset


def test_merge_world_with_dataset_left_merge_keeps_all_world_rows() -> None:
    world = gpd.GeoDataFrame(
        {"ISO_A3": ["PRT", "ESP"], "geometry": [Point(0, 0), Point(1, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Mimics OWID Grapher: Entity, Code, Year, <metric>
    df = pd.DataFrame(
        {
            "Entity": ["Portugal", "Portugal"],
            "Code": ["PRT", "PRT"],
            "Year": [2020, 2021],
            "annual_deforestation": [1.0, 2.0],
        }
    )

    merged = merge_world_with_dataset(world, df, latest_year_only=True)

    assert isinstance(merged, gpd.GeoDataFrame)
    assert len(merged) == 2                      # left join keeps both
    assert "geometry" in merged.columns

    # latest year is 2021, so Portugal should match 2.0
    prt_val = merged.loc[merged["ISO_A3"] == "PRT", "annual_deforestation"].iloc[0]
    esp_val = merged.loc[merged["ISO_A3"] == "ESP", "annual_deforestation"].iloc[0]

    assert prt_val == 2.0
    assert pd.isna(esp_val)