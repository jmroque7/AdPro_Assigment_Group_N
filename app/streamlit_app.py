from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Import your existing utilities/class
from okavango import OkavangoProject, OkavangoConfig


# ---------------------------
# App configuration (pink theme)
# ---------------------------
st.set_page_config(
    page_title="Project Okavango • Environmental Data Maps",
    page_icon="🗺️",
    layout="wide",
)

PINK_CMAP = "RdPu"  # pink colormap


@dataclass(frozen=True)
class DatasetInfo:
    filename: str
    label: str


DATASETS: List[DatasetInfo] = [
    DatasetInfo("annual-change-forest-area.csv", "Annual change in forest area"),
    DatasetInfo("annual-deforestation.csv", "Annual deforestation"),
    DatasetInfo("terrestrial-protected-areas.csv", "Share of land protected"),
    DatasetInfo("share-degraded-land.csv", "Share of degraded land"),
    DatasetInfo("forest-area-as-share-of-land-area.csv", "Forest area as share of land area"),
]


# ---------------------------
# Helpers
# ---------------------------
def detect_metric_column(df: pd.DataFrame) -> str:
    """Pick the dataset value column, ignoring OWID meta columns and annotation columns."""
    meta = {"Entity", "Code", "Year", "iso3"}
    candidates = [
        c for c in df.columns
        if c not in meta and "annotation" not in c.lower()
    ]
    if not candidates:
        raise ValueError("Could not detect metric column in dataset.")
    return candidates[-1]


def year_options(df: pd.DataFrame) -> List[int]:
    years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique()
    return sorted(years.tolist())


def filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out = out[out["Year"] == year]
    return out


def clean_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only valid ISO3 rows and standardize column name to iso3."""
    out = df.copy()
    out = out.rename(columns={"Code": "iso3"})
    out["iso3"] = out["iso3"].astype(str)
    out = out[(out["iso3"] != "nan") & (out["iso3"].str.len() == 3)]
    return out


def merge_world_left(world: gpd.GeoDataFrame, df_year: pd.DataFrame) -> gpd.GeoDataFrame:
    """Teacher rule: left dataframe is the GeoDataFrame."""
    world_gdf = world.copy()

    # Find ISO column in Natural Earth
    iso_candidates = ["ISO_A3", "ADM0_A3", "ISO_A3_EH", "ADM0_A3_US"]
    iso_col = next((c for c in iso_candidates if c in world_gdf.columns), None)
    if iso_col is None:
        raise ValueError("Natural Earth world dataset missing ISO3 column.")

    world_gdf["iso3"] = world_gdf[iso_col].astype(str)

    # Make sure right side has iso3
    df_year = df_year.copy()
    if "iso3" not in df_year.columns:
        raise ValueError("Dataset missing iso3 column after cleaning.")

    # One row per country
    df_year = df_year.drop_duplicates(subset=["iso3"])

    merged = world_gdf.merge(df_year, on="iso3", how="left")
    return merged


def top_bottom(df: pd.DataFrame, metric_col: str, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    temp = df[["Entity", metric_col]].dropna().copy()
    temp = temp.sort_values(metric_col, ascending=False)
    top = temp.head(n)
    bottom = temp.tail(n).sort_values(metric_col, ascending=True)
    return top, bottom


# ---------------------------
# Load project (cache to be fast)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_project() -> Tuple[gpd.GeoDataFrame, Dict[str, pd.DataFrame]]:
    """
    Load OkavangoProject once.
    - Uses your class to ensure downloads happen (if possible).
    - Uses the class datasets dict (raw dataframes).
    """
    cfg = OkavangoConfig(project_root=Path("."), download=True, latest_year_only=False)
    proj = OkavangoProject(cfg)
    return proj.world, proj.datasets


# ---------------------------
# UI
# ---------------------------
st.markdown(
    """
    <style>
      .pink-title { color: #ff4fa3; font-weight: 800; }
      .pink-sub { color: #ff4fa3; opacity: 0.9; }
      div[data-testid="stMetricValue"] { color: #ff4fa3; }
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="pink-title" style="font-size:34px;">Project Okavango</div>', unsafe_allow_html=True)
st.markdown('<div class="pink-sub" style="font-size:16px;">Interactive maps + analytics using the most recent data available (and any year you choose)</div>', unsafe_allow_html=True)
st.write("")

with st.spinner("Loading world map and datasets..."):
    world_gdf, datasets = load_project()

# Sidebar controls
st.sidebar.markdown("## 🎛️ Controls")
dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    options=[d.label for d in DATASETS],
)

dataset_filename = next(d.filename for d in DATASETS if d.label == dataset_choice)
df_raw = datasets[dataset_filename]
metric_col = detect_metric_column(df_raw)

years = year_options(df_raw)
latest_year = max(years) if years else None

year_choice = st.sidebar.selectbox(
    "Choose year",
    options=years,
    index=len(years) - 1 if years else 0
)

show_labels = st.sidebar.toggle("Show country labels (slower)", value=False)
st.sidebar.caption("Tip: labels can be slow. Keep off for performance.")

# Filter + clean
df_year = filter_year(df_raw, year_choice)
df_year = clean_iso3(df_year)

# Merge (GeoDF left!)
merged = merge_world_left(world_gdf, df_year)

# Compute metrics
n_total_countries = len(world_gdf)
n_with_data = int(merged[metric_col].notna().sum()) if metric_col in merged.columns else 0
missing_pct = round((1 - (n_with_data / n_total_countries)) * 100, 2) if n_total_countries else 0

top5, bottom5 = top_bottom(df_year, metric_col, n=5)

# ---------------------------
# Layout: Map + KPIs
# ---------------------------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("🗺️ World map")

    fig, ax = plt.subplots(figsize=(13, 7))

    merged.plot(
        column=metric_col,
        cmap=PINK_CMAP,
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    ax.set_axis_off()
    ax.set_title(f"{dataset_choice} — {year_choice}", fontsize=16)

    if show_labels:
        # Light labels (optional)
        centroids = merged.copy()
        centroids["centroid"] = centroids.geometry.centroid
        for _, row in centroids.dropna(subset=[metric_col]).head(40).iterrows():
            x, y = row["centroid"].x, row["centroid"].y
            ax.text(x, y, str(row.get("NAME", ""))[:10], fontsize=6, alpha=0.5)

    st.pyplot(fig, use_container_width=True)

with right:
    st.subheader("📌 Key metrics")

    c1, c2 = st.columns(2)
    c1.metric("Selected year", f"{year_choice}")
    c2.metric("Latest available", f"{latest_year}")

    c3, c4 = st.columns(2)
    c3.metric("Countries with data", f"{n_with_data}/{n_total_countries}")
    c4.metric("Missing countries", f"{missing_pct}%")

    st.markdown("### What you’re looking at")
    st.write(
        "The map shows the selected indicator for the selected year. "
        "Grey countries have no data for that year. "
        "Below you’ll find top/bottom countries and a distribution chart."
    )

# ---------------------------
# Bottom: Top/Bottom + Charts
# ---------------------------
st.write("")
st.subheader("📊 Analysis")

colA, colB = st.columns([1, 1])

with colA:
    st.markdown("### 🔝 Top 5 countries")
    st.dataframe(top5.rename(columns={metric_col: "value"}), use_container_width=True)

    fig_top, ax_top = plt.subplots(figsize=(7, 4))
    ax_top.barh(top5["Entity"], top5[metric_col])
    ax_top.invert_yaxis()
    ax_top.set_title("Top 5 (highest values)")
    ax_top.set_xlabel("Value")
    st.pyplot(fig_top, use_container_width=True)

with colB:
    st.markdown("### 🔻 Bottom 5 countries")
    st.dataframe(bottom5.rename(columns={metric_col: "value"}), use_container_width=True)

    fig_bot, ax_bot = plt.subplots(figsize=(7, 4))
    ax_bot.barh(bottom5["Entity"], bottom5[metric_col])
    ax_bot.invert_yaxis()
    ax_bot.set_title("Bottom 5 (lowest values)")
    ax_bot.set_xlabel("Value")
    st.pyplot(fig_bot, use_container_width=True)

st.write("")
st.markdown("### Distribution")
vals = df_year[metric_col].dropna()

fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
ax_hist.hist(vals, bins=30)
ax_hist.set_title("Histogram of values across countries")
ax_hist.set_xlabel("Value")
ax_hist.set_ylabel("Number of countries")
st.pyplot(fig_hist, use_container_width=True)

st.caption(
    "Note: Values come from OWID grapher datasets and Natural Earth country polygons. "
    "If a country has no ISO3 match or missing data for a year, it appears grey."
)