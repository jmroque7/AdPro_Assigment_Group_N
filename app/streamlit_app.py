from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from app.ai_workflow import (
    DEFAULT_MODELS_CONFIG_PATH,
    OllamaError,
    WorkflowConfigError,
    execute_governed_ai_workflow,
    get_ollama_base_url,
    load_workflow_config,
    list_ollama_models,
    workflow_cache_key,
)
from app.okavango import OkavangoConfig, OkavangoProject


st.set_page_config(
    page_title="Project Okavango",
    page_icon="map",
    layout="wide",
)

PINK_CMAP = "RdPu"
IMAGES_DIR = Path("images")
MODELS_CONFIG_PATH = DEFAULT_MODELS_CONFIG_PATH
ZOOM_LEVELS = {
    "Continent view": 4,
    "Country view": 6,
    "Region view": 8,
    "City view": 10,
    "District view": 12,
    "Neighborhood view": 14,
    "Street area": 16,
}
COUNTRY_CITY_OPTIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Australia": {
        "Brisbane": (-27.4698, 153.0251),
        "Darwin": (-12.4634, 130.8456),
        "Melbourne": (-37.8136, 144.9631),
        "Perth": (-31.9505, 115.8605),
        "Sydney": (-33.8688, 151.2093),
    },
    "Botswana": {
        "Gaborone": (-24.6282, 25.9231),
        "Maun": (-19.9833, 23.4162),
        "Kasane": (-17.8167, 25.1500),
        "Okavango Delta": (-19.3000, 22.9000),
    },
    "Brazil": {
        "Brasilia": (-15.7939, -47.8828),
        "Manaus": (-3.1190, -60.0217),
        "Sao Paulo": (-23.5505, -46.6333),
        "Rio de Janeiro": (-22.9068, -43.1729),
    },
    "Democratic Republic of the Congo": {
        "Goma": (-1.6792, 29.2228),
        "Kinshasa": (-4.4419, 15.2663),
        "Lubumbashi": (-11.6647, 27.4794),
    },
    "Egypt": {
        "Alexandria": (31.2001, 29.9187),
        "Cairo": (30.0444, 31.2357),
        "Luxor": (25.6872, 32.6396),
    },
    "India": {
        "Bengaluru": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Mumbai": (19.0760, 72.8777),
        "New Delhi": (28.6139, 77.2090),
    },
    "Kenya": {
        "Mombasa": (-4.0435, 39.6682),
        "Nairobi": (-1.2864, 36.8172),
        "Turkana": (3.1167, 35.6000),
    },
    "Mozambique": {
        "Beira": (-19.8333, 34.8500),
        "Gorongosa": (-18.6800, 34.0800),
        "Maputo": (-25.9692, 32.5732),
        "Pemba": (-12.9730, 40.5178),
    },
    "Namibia": {
        "Etosha": (-18.7750, 16.8825),
        "Swakopmund": (-22.6784, 14.5266),
        "Windhoek": (-22.5609, 17.0658),
    },
    "Portugal": {
        "Coimbra": (40.2033, -8.4103),
        "Faro": (37.0194, -7.9304),
        "Lisbon": (38.7223, -9.1393),
        "Porto": (41.1579, -8.6291),
    },
    "South Africa": {
        "Cape Town": (-33.9249, 18.4241),
        "Johannesburg": (-26.2041, 28.0473),
        "Kruger National Park": (-24.0000, 31.5000),
        "Durban": (-29.8587, 31.0218),
        "Pretoria": (-25.7479, 28.2293),
    },
    "Spain": {
        "Barcelona": (41.3874, 2.1686),
        "Seville": (37.3891, -5.9845),
        "Valencia": (39.4699, -0.3763),
        "Madrid": (40.4168, -3.7038),
    },
    "Tanzania": {
        "Dar es Salaam": (-6.7924, 39.2083),
        "Dodoma": (-6.1630, 35.7516),
        "Serengeti": (-2.3333, 34.8333),
    },
    "United Kingdom": {
        "Edinburgh": (55.9533, -3.1883),
        "London": (51.5072, -0.1276),
        "Manchester": (53.4808, -2.2426),
    },
    "United States": {
        "Los Angeles": (34.0522, -118.2437),
        "New York": (40.7128, -74.0060),
        "San Francisco": (37.7749, -122.4194),
        "Yellowstone": (44.4280, -110.5885),
    },
    "Zambia": {
        "Lusaka": (-15.3875, 28.3228),
        "South Luangwa": (-13.1167, 31.7833),
        "Victoria Falls": (-17.9243, 25.8572),
    },
    "Zimbabwe": {
        "Harare": (-17.8252, 31.0335),
        "Hwange": (-18.6299, 26.9536),
        "Victoria Falls": (-17.9244, 25.8567),
    },
}


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

DESCRIPTIONS = {
    "Annual change in forest area": "Net yearly change in forest coverage across countries.",
    "Annual deforestation": "Amount of forest area lost per year.",
    "Share of land protected": "Percentage of land area that is legally protected.",
    "Share of degraded land": "Percentage of land affected by degradation.",
    "Forest area as share of land area": "Forest coverage relative to total land area.",
}

RISK_COLOR = {
    "low": "#2b9348",
    "medium": "#f48c06",
    "high": "#d00000",
}

PRIMARY_COLOR = "#ff4fa3"
NEGATIVE_COLOR = "#ff6b6b"
CHART_BG = "#111827"
GRID_COLOR = "#2a3346"


def detect_metric_column(df: pd.DataFrame) -> str:
    meta = {"Entity", "Code", "Year", "iso3"}
    candidates = [c for c in df.columns if c not in meta and "annotation" not in c.lower()]
    if not candidates:
        raise ValueError("Could not detect metric column in dataset.")
    return candidates[-1]


def year_options(df: pd.DataFrame) -> List[int]:
    years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique()
    return sorted(years.tolist())


def filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    return out[out["Year"] == year]


def clean_iso3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(columns={"Code": "iso3"})
    out["iso3"] = out["iso3"].astype(str)
    return out[(out["iso3"] != "nan") & (out["iso3"].str.len() == 3)]


def merge_world_left(world: gpd.GeoDataFrame, df_year: pd.DataFrame) -> gpd.GeoDataFrame:
    world_gdf = world.copy()
    iso_candidates = ["ISO_A3", "ADM0_A3", "ISO_A3_EH", "ADM0_A3_US"]
    iso_col = next((c for c in iso_candidates if c in world_gdf.columns), None)
    if iso_col is None:
        raise ValueError("Natural Earth world dataset missing ISO3 column.")

    world_gdf["iso3"] = world_gdf[iso_col].astype(str)
    df_year = df_year.drop_duplicates(subset=["iso3"])
    return world_gdf.merge(df_year, on="iso3", how="left")


def top_bottom(df: pd.DataFrame, metric_col: str, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    temp = df[["Entity", metric_col]].dropna().copy().sort_values(metric_col, ascending=False)
    return temp.head(n), temp.tail(n).sort_values(metric_col, ascending=True)


def format_compact_number(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"

    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.{decimals}f}k"
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def format_metric_value(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    magnitude = abs(float(value))
    if magnitude >= 1000:
        return format_compact_number(float(value), decimals=2)
    if magnitude >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def format_delta(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "No prior-year match"
    sign = "+" if value > 0 else ""
    return f"{sign}{format_metric_value(float(value))} vs previous year"


def style_chart_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors="#dbe4f3", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#364055")
    ax.xaxis.label.set_color("#dbe4f3")
    ax.yaxis.label.set_color("#dbe4f3")
    ax.title.set_color("#f7fafc")
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


def add_card_container_start(class_name: str = "ok-section-card") -> None:
    st.markdown(f'<div class="{class_name}">', unsafe_allow_html=True)


def add_card_container_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_panel(label: str, value: str, detail: str = "", tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="ok-metric-panel ok-metric-{tone}">
            <div class="ok-metric-label">{label}</div>
            <div class="ok-metric-value">{value}</div>
            <div class="ok-metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_banner(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ok-hero-card">
            <div class="ok-hero-eyebrow">Dashboard insight</div>
            <div class="ok-hero-title">{title}</div>
            <div class="ok-hero-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_continent_summary(merged: gpd.GeoDataFrame, metric_col: str) -> pd.DataFrame:
    if "CONTINENT" not in merged.columns:
        return pd.DataFrame(columns=["CONTINENT", "median_value", "country_count"])

    continent_df = merged.dropna(subset=[metric_col]).copy()
    if continent_df.empty:
        return pd.DataFrame(columns=["CONTINENT", "median_value", "country_count"])

    summary = (
        continent_df.groupby("CONTINENT", dropna=True)[metric_col]
        .agg(median_value="median", country_count="count")
        .reset_index()
        .sort_values("median_value", ascending=False)
    )
    return summary


def compute_dashboard_summary(
    df_raw: pd.DataFrame,
    df_year: pd.DataFrame,
    merged: gpd.GeoDataFrame,
    metric_col: str,
    year_choice: int,
) -> Dict[str, object]:
    values = df_year[metric_col].dropna()
    previous_years = [year for year in year_options(df_raw) if year < year_choice]
    previous_year = previous_years[-1] if previous_years else None
    previous_df = clean_iso3(filter_year(df_raw, previous_year)) if previous_year is not None else pd.DataFrame()

    year_over_year_delta: Optional[float] = None
    if not previous_df.empty:
        paired = (
            df_year[["iso3", metric_col]]
            .dropna()
            .merge(
                previous_df[["iso3", metric_col]].dropna(),
                on="iso3",
                how="inner",
                suffixes=("_current", "_previous"),
            )
        )
        if not paired.empty:
            year_over_year_delta = float(
                paired[f"{metric_col}_current"].median() - paired[f"{metric_col}_previous"].median()
            )

    positive_count = int((values > 0).sum())
    negative_count = int((values < 0).sum())
    neutral_count = int((values == 0).sum())
    total_non_null = int(values.shape[0])

    top_row = df_year[["Entity", metric_col]].dropna().sort_values(metric_col, ascending=False).head(1)
    bottom_row = df_year[["Entity", metric_col]].dropna().sort_values(metric_col, ascending=True).head(1)

    return {
        "count": total_non_null,
        "median": float(values.median()) if not values.empty else None,
        "mean": float(values.mean()) if not values.empty else None,
        "q1": float(values.quantile(0.25)) if not values.empty else None,
        "q3": float(values.quantile(0.75)) if not values.empty else None,
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if not values.empty else None,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_share": (positive_count / total_non_null * 100) if total_non_null else 0.0,
        "negative_share": (negative_count / total_non_null * 100) if total_non_null else 0.0,
        "top_country": top_row.iloc[0]["Entity"] if not top_row.empty else "n/a",
        "top_value": float(top_row.iloc[0][metric_col]) if not top_row.empty else None,
        "bottom_country": bottom_row.iloc[0]["Entity"] if not bottom_row.empty else "n/a",
        "bottom_value": float(bottom_row.iloc[0][metric_col]) if not bottom_row.empty else None,
        "previous_year": previous_year,
        "year_over_year_delta": year_over_year_delta,
        "continent_summary": compute_continent_summary(merged, metric_col),
    }


@st.cache_data(show_spinner=False)
def load_project() -> Tuple[gpd.GeoDataFrame, Dict[str, pd.DataFrame]]:
    cfg = OkavangoConfig(project_root=Path("."), download=True, latest_year_only=False)
    project = OkavangoProject(cfg)
    return project.world, project.datasets


def render_header() -> None:
    st.markdown(
        """
        <style>
          :root {
              --ok-card-border: rgba(255, 255, 255, 0.09);
              --ok-card-bg: linear-gradient(180deg, rgba(18,25,38,0.98), rgba(12,17,27,0.96));
              --ok-muted: #b8bcc8;
              --ok-accent: #ff4fa3;
              --ok-accent-soft: rgba(255, 79, 163, 0.14);
              --ok-surface: #101722;
              --ok-surface-2: #121c2a;
              --ok-green-soft: rgba(47, 191, 113, 0.16);
              --ok-red-soft: rgba(255, 107, 107, 0.16);
          }
          .stApp {
              background:
                  radial-gradient(circle at top right, rgba(255,79,163,0.14), transparent 28%),
                  radial-gradient(circle at top left, rgba(61,169,252,0.10), transparent 24%),
                  linear-gradient(180deg, #0a0f18 0%, #0f141d 100%);
          }
          .ok-title { color: #ff4fa3; font-weight: 800; font-size: 34px; letter-spacing: -0.02em; }
          .ok-sub { color: #c18ab0; font-size: 16px; }
          .ok-section-card {
              border: 1px solid var(--ok-card-border);
              border-radius: 18px;
              padding: 1rem 1rem 0.75rem 1rem;
              margin-bottom: 1rem;
              background: var(--ok-card-bg);
              box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
          }
          .ok-hero-card {
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 22px;
              padding: 1.15rem 1.2rem;
              margin: 0.35rem 0 1rem 0;
              background:
                  linear-gradient(135deg, rgba(22,38,62,0.96), rgba(12,18,28,0.95)),
                  linear-gradient(135deg, rgba(255,79,163,0.06), rgba(61,169,252,0.06));
              box-shadow: 0 18px 36px rgba(0, 0, 0, 0.2);
          }
          .ok-hero-eyebrow {
              color: #8cb8ff;
              text-transform: uppercase;
              letter-spacing: 0.12em;
              font-size: 0.72rem;
              margin-bottom: 0.35rem;
          }
          .ok-hero-title {
              color: #f5f7fb;
              font-size: 1.25rem;
              font-weight: 700;
              margin-bottom: 0.25rem;
          }
          .ok-hero-text {
              color: #b6c2d6;
              font-size: 0.95rem;
              line-height: 1.55;
          }
          .ok-section-title {
              font-size: 1rem;
              font-weight: 700;
              margin-bottom: 0.25rem;
              letter-spacing: 0.01em;
          }
          .ok-section-text {
              color: var(--ok-muted);
              font-size: 0.92rem;
              margin-bottom: 0.8rem;
          }
          .ok-results-metric {
              border: 1px solid var(--ok-card-border);
              background: rgba(255, 255, 255, 0.025);
              border-radius: 14px;
              padding: 0.7rem 0.85rem;
              margin-bottom: 0.55rem;
          }
          .ok-results-label {
              font-size: 0.78rem;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: var(--ok-muted);
              margin-bottom: 0.18rem;
          }
          .ok-results-value {
              font-size: 0.98rem;
              font-weight: 600;
              line-height: 1.35;
          }
          .ok-page-note {
              border-left: 3px solid var(--ok-accent);
              padding: 0.75rem 0.9rem;
              border-radius: 0 12px 12px 0;
              background: var(--ok-accent-soft);
              margin: 0.5rem 0 1rem 0;
          }
          .ok-metric-panel {
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 18px;
              padding: 0.9rem 1rem;
              background: linear-gradient(180deg, rgba(18, 26, 39, 0.96), rgba(11, 16, 25, 0.94));
              box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
              min-height: 122px;
          }
          .ok-metric-positive { background: linear-gradient(180deg, rgba(18, 35, 28, 0.96), rgba(11, 20, 17, 0.94)); }
          .ok-metric-negative { background: linear-gradient(180deg, rgba(38, 20, 28, 0.96), rgba(20, 11, 16, 0.94)); }
          .ok-metric-highlight { background: linear-gradient(180deg, rgba(33, 26, 48, 0.96), rgba(14, 13, 24, 0.94)); }
          .ok-metric-label {
              font-size: 0.78rem;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #8fa3bf;
              margin-bottom: 0.45rem;
          }
          .ok-metric-value {
              font-size: 1.8rem;
              font-weight: 750;
              color: #f8fbff;
              line-height: 1.05;
              margin-bottom: 0.4rem;
          }
          .ok-metric-detail {
              color: #b8c5d8;
              font-size: 0.88rem;
              line-height: 1.45;
          }
          div[data-testid="stMetric"] {
              background: linear-gradient(180deg, rgba(18, 25, 38, 0.96), rgba(12, 17, 27, 0.94));
              border: 1px solid rgba(255,255,255,0.08);
              padding: 0.9rem 1rem;
              border-radius: 16px;
          }
          div[data-testid="stMetricLabel"] {
              color: #9fb0c7;
          }
          div[data-testid="stMetricValue"] {
              color: #f7fafc;
          }
          div[data-testid="stDataFrame"] {
              border-radius: 16px;
              overflow: hidden;
              border: 1px solid rgba(255,255,255,0.08);
          }
          .risk-card {
              border-radius: 18px;
              padding: 1rem 1.2rem;
              color: white;
              font-weight: 700;
              box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          }
          .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ok-title">Project Okavango</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ok-sub">Environmental analytics with OWID data, ESRI imagery, and local AI triage.</div>',
        unsafe_allow_html=True,
    )
    st.write("")


def render_map_page(world_gdf: gpd.GeoDataFrame, datasets: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Global Environmental Analytics")

    dataset_choice = st.sidebar.selectbox(
        "Choose dataset",
        options=[d.label for d in DATASETS],
        key="dataset_choice",
    )
    dataset_filename = next(d.filename for d in DATASETS if d.label == dataset_choice)
    df_raw = datasets[dataset_filename]
    metric_col = detect_metric_column(df_raw)

    years = year_options(df_raw)
    latest_year = max(years) if years else None
    year_choice = st.sidebar.selectbox(
        "Choose year",
        options=years,
        index=len(years) - 1 if years else 0,
        key="year_choice",
    )
    show_labels = st.sidebar.toggle("Show country labels", value=False, key="show_labels")

    st.info(DESCRIPTIONS[dataset_choice])

    df_year = clean_iso3(filter_year(df_raw, year_choice))
    merged = merge_world_left(world_gdf, df_year)

    n_total_countries = len(world_gdf)
    n_with_data = int(merged[metric_col].notna().sum()) if metric_col in merged.columns else 0
    missing_pct = round((1 - (n_with_data / n_total_countries)) * 100, 2) if n_total_countries else 0
    top5, bottom5 = top_bottom(df_year, metric_col, n=5)
    dashboard_summary = compute_dashboard_summary(df_raw, df_year, merged, metric_col, year_choice)

    render_insight_banner(
        title=f"{dataset_choice} in {year_choice}",
        body=(
            f"{DESCRIPTIONS[dataset_choice]} Median country value is "
            f"{format_metric_value(dashboard_summary['median'])}, with "
            f"{dashboard_summary['positive_share']:.0f}% of reporting countries above zero "
            f"and {dashboard_summary['negative_share']:.0f}% below zero."
        ),
    )

    headline_cols = st.columns(4)
    with headline_cols[0]:
        render_metric_panel(
            "Coverage",
            f"{n_with_data}/{n_total_countries}",
            f"{missing_pct}% of countries are unmatched for the selected year.",
            tone="highlight",
        )
    with headline_cols[1]:
        render_metric_panel(
            "Median value",
            format_metric_value(dashboard_summary["median"]),
            format_delta(dashboard_summary["year_over_year_delta"]),
        )
    with headline_cols[2]:
        render_metric_panel(
            "Highest country",
            dashboard_summary["top_country"],
            format_metric_value(dashboard_summary["top_value"]),
            tone="positive",
        )
    with headline_cols[3]:
        render_metric_panel(
            "Lowest country",
            dashboard_summary["bottom_country"],
            format_metric_value(dashboard_summary["bottom_value"]),
            tone="negative",
        )

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("### World map")
        add_card_container_start()
        fig, ax = plt.subplots(figsize=(13, 7), facecolor=CHART_BG)
        merged.plot(
            column=metric_col,
            cmap=PINK_CMAP,
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="#253041",
            linewidth=0.35,
        )
        ax.set_axis_off()
        ax.set_title(f"{dataset_choice} - {year_choice}", fontsize=16, color="#f8fbff", pad=14)

        cbar_ax = fig.axes[-1]
        cbar_ax.set_facecolor(CHART_BG)
        cbar_ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        cbar_ax.tick_params(colors="#dbe4f3")
        cbar_ax.yaxis.label.set_color("#dbe4f3")

        if show_labels:
            centroids = merged.copy()
            centroids["centroid"] = centroids.geometry.centroid
            for _, row in centroids.dropna(subset=[metric_col]).head(40).iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row.get("NAME", ""))[:10],
                    fontsize=6,
                    alpha=0.55,
                    color="#dce7f7",
                )
        st.pyplot(fig, use_container_width=True)
        add_card_container_end()

    with right:
        st.markdown("### Key metrics")
        add_card_container_start()
        c1, c2 = st.columns(2)
        c1.metric("Selected year", f"{year_choice}")
        c2.metric("Latest available", f"{latest_year}")
        c3, c4 = st.columns(2)
        c3.metric("Countries with data", f"{n_with_data}/{n_total_countries}")
        c4.metric("Missing countries", f"{missing_pct}%")
        c5, c6 = st.columns(2)
        c5.metric("Positive values", f"{dashboard_summary['positive_count']}")
        c6.metric("Negative values", f"{dashboard_summary['negative_count']}")
        st.caption(
            "Grey countries have no matched data for the selected year. "
            "The median delta compares the selected year with the previous year on countries that report in both periods."
        )
        add_card_container_end()

        add_card_container_start()
        st.markdown("#### Distribution at a glance")
        st.caption(
            f"Middle 50% of countries range from {format_metric_value(dashboard_summary['q1'])} "
            f"to {format_metric_value(dashboard_summary['q3'])}. "
            f"That gives an interquartile spread of {format_metric_value(dashboard_summary['iqr'])}."
        )
        render_metric_panel(
            "Balance",
            f"{dashboard_summary['positive_share']:.0f}% / {dashboard_summary['negative_share']:.0f}%",
            "Positive vs negative share across reporting countries.",
            tone="highlight",
        )
        add_card_container_end()

    st.markdown("### Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        add_card_container_start()
        st.markdown("#### Top 5 countries")
        st.dataframe(
            top5.rename(columns={metric_col: "value"}).style.format({"value": format_metric_value}),
            use_container_width=True,
        )
        if top5.empty:
            st.info("No country-level records are available for this year.")
        else:
            fig_top, ax_top = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
            ax_top.barh(top5["Entity"], top5[metric_col], color=PRIMARY_COLOR, alpha=0.92)
            ax_top.invert_yaxis()
            ax_top.set_xlabel("Value")
            ax_top.set_title("Highest country values", pad=10)
            style_chart_axis(ax_top)
            st.pyplot(fig_top, use_container_width=True)
        add_card_container_end()

    with col_b:
        add_card_container_start()
        st.markdown("#### Bottom 5 countries")
        st.dataframe(
            bottom5.rename(columns={metric_col: "value"}).style.format({"value": format_metric_value}),
            use_container_width=True,
        )
        if bottom5.empty:
            st.info("No country-level records are available for this year.")
        else:
            fig_bottom, ax_bottom = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
            ax_bottom.barh(bottom5["Entity"], bottom5[metric_col], color=NEGATIVE_COLOR, alpha=0.9)
            ax_bottom.invert_yaxis()
            ax_bottom.set_xlabel("Value")
            ax_bottom.set_title("Lowest country values", pad=10)
            style_chart_axis(ax_bottom)
            st.pyplot(fig_bottom, use_container_width=True)
        add_card_container_end()

    st.markdown("### Distribution and regional signal")
    distribution_col, region_col = st.columns([1.7, 1])
    values = df_year[metric_col].dropna()
    with distribution_col:
        add_card_container_start()
        if values.empty:
            st.info("No values are available to build a distribution plot for this year.")
        else:
            fig_hist, ax_hist = plt.subplots(figsize=(12, 4), facecolor=CHART_BG)
            ax_hist.hist(values, bins=30, alpha=0.82, color="#66b3ff", edgecolor="#c9e6ff", linewidth=0.5)
            ax_hist.axvline(values.median(), color=PRIMARY_COLOR, linestyle="--", linewidth=1.8, label="Median")
            ax_hist.set_title("Histogram of values across countries")
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Number of countries")
            style_chart_axis(ax_hist)
            ax_hist.legend(facecolor=CHART_BG, edgecolor="#364055", labelcolor="#eaf2ff")
            st.pyplot(fig_hist, use_container_width=True)
        add_card_container_end()

    with region_col:
        add_card_container_start()
        st.markdown("#### By continent")
        continent_summary = dashboard_summary["continent_summary"]
        if isinstance(continent_summary, pd.DataFrame) and not continent_summary.empty:
            fig_region, ax_region = plt.subplots(figsize=(6, 4.2), facecolor=CHART_BG)
            ax_region.barh(
                continent_summary["CONTINENT"],
                continent_summary["median_value"],
                color="#7cc6fe",
                alpha=0.9,
            )
            ax_region.invert_yaxis()
            ax_region.set_xlabel("Median value")
            ax_region.set_title("Median by continent", pad=10)
            style_chart_axis(ax_region)
            st.pyplot(fig_region, use_container_width=True)
            st.dataframe(
                continent_summary.rename(
                    columns={
                        "CONTINENT": "continent",
                        "median_value": "median value",
                        "country_count": "countries",
                    }
                ).style.format({"median value": format_metric_value}),
                use_container_width=True,
            )
        else:
            st.info("Continent-level summary is not available for this map source.")
        add_card_container_end()


def render_risk_badge(risk_level: str, flagged: bool, score: int) -> None:
    color = RISK_COLOR.get(risk_level, "#577590")
    label = "FLAGGED" if flagged else "NOT FLAGGED"
    st.markdown(
        f"""
        <div class="risk-card" style="background:{color};">
            Environmental risk: {risk_level.upper()} ({score}/100) - {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_compat(image_path: Path, caption: str) -> None:
    try:
        st.image(str(image_path), caption=caption, use_container_width=True)
    except TypeError:
        st.image(str(image_path), caption=caption)


def zoom_label_from_value(zoom: int) -> str:
    for label, value in ZOOM_LEVELS.items():
        if value == zoom:
            return label
    return f"Custom zoom ({zoom})"


def open_section_card(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="ok-section-card">
            <div class="ok-section-title">{title}</div>
            <div class="ok-section-text">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def close_section_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_fact(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="ok-results-metric">
            <div class="ok-results-label">{label}</div>
            <div class="ok-results-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_page() -> None:
    st.subheader("AI Workflow: Image-Based Environmental Risk Check")
    st.write(
        "Choose an area of interest, download recent ESRI World Imagery for that location, generate a satellite-image description with a local vision model, and then run a second model to identify potential environmental risk."
    )
    st.markdown(
        """
        <div class="ok-page-note">
            This workflow is designed for quick environmental screening: first the image is described, then the description is translated into a structured risk assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models: list[str] = []
    try:
        available_models = list_ollama_models()
    except OllamaError:
        available_models = []
    try:
        workflow_config = load_workflow_config(MODELS_CONFIG_PATH)
    except WorkflowConfigError as exc:
        st.error(str(exc))
        st.info("Create or fix `models.yaml` in the project root to run the governed AI workflow.")
        return

    open_section_card(
        "1. Choose How To Select A Location",
        "Use the built-in country and city lists for a quick demo, or enter coordinates manually for a custom area.",
    )
    location_mode = st.radio(
        "Location mode",
        options=["Country and city", "Manual coordinates"],
        key="ai_location_mode",
        label_visibility="collapsed",
    )
    close_section_card()

    countries = sorted(COUNTRY_CITY_OPTIONS)
    default_country = "Botswana" if "Botswana" in COUNTRY_CITY_OPTIONS else countries[0]
    if "ai_selected_country" not in st.session_state:
        st.session_state["ai_selected_country"] = default_country
    if st.session_state["ai_selected_country"] not in COUNTRY_CITY_OPTIONS:
        st.session_state["ai_selected_country"] = default_country

    if location_mode == "Country and city":
        open_section_card(
            "2. Select A Country And City",
            "Pick a country first, then choose one of its available cities or landmarks.",
        )
        location_col1, location_col2 = st.columns(2)
        previous_country = st.session_state["ai_selected_country"]
        country = location_col1.selectbox(
            "Country",
            options=countries,
            index=countries.index(previous_country),
            key="ai_selected_country",
        )
        cities = list(COUNTRY_CITY_OPTIONS[country].keys())
        default_city = "Maun" if country == "Botswana" and "Maun" in cities else cities[0]
        if st.session_state.get("ai_selected_city") not in cities or previous_country != country:
            st.session_state["ai_selected_city"] = default_city
        city = location_col2.selectbox(
            "City",
            options=cities,
            index=cities.index(st.session_state["ai_selected_city"]),
            key="ai_selected_city",
        )
        latitude, longitude = COUNTRY_CITY_OPTIONS[country][city]
        location_label = f"{city}, {country}"
        st.caption(f"Selected coordinates: {latitude:.4f}, {longitude:.4f}")
        close_section_card()
    else:
        open_section_card(
            "2. Enter Coordinates",
            "Provide latitude and longitude to analyze a custom area outside the built-in location list.",
        )
        country = ""
        city = ""
        location_label = "Custom coordinates"
        col1, col2 = st.columns(2)
        latitude = col1.number_input(
            "Latitude",
            min_value=-85.0,
            max_value=85.0,
            value=float(st.session_state.get("ai_manual_latitude", -19.0)),
            step=0.1,
            key="ai_manual_latitude",
        )
        longitude = col2.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.get("ai_manual_longitude", 23.0)),
            step=0.1,
            key="ai_manual_longitude",
        )
        st.caption(f"Selected coordinates: {latitude:.4f}, {longitude:.4f}")
        close_section_card()

    open_section_card(
        "3. Configure The Analysis",
        "Choose the satellite view scale. The AI models, prompts, and settings come directly from models.yaml for reproducibility.",
    )
    zoom_choice = st.select_slider(
        "Zoom level",
        options=list(ZOOM_LEVELS.keys()),
        value=st.session_state.get("ai_zoom_choice", "City view"),
        help="Choose how wide the satellite image should be around the selected point.",
        key="ai_zoom_choice",
    )
    zoom = ZOOM_LEVELS[zoom_choice]
    st.caption(f"Current zoom: {zoom_choice} ({zoom})")
    col4, col5 = st.columns(2)
    col4.caption(f"Image model: {workflow_config.image_analysis.model}")
    col5.caption(f"Text model: {workflow_config.text_analysis.model}")
    st.caption(
        "Image settings: "
        f"{json.dumps(workflow_config.image_analysis.settings, sort_keys=True)}"
    )
    st.caption(
        "Text settings: "
        f"{json.dumps(workflow_config.text_analysis.settings, sort_keys=True)}"
    )
    with st.expander("Configured prompts from models.yaml"):
        st.markdown("**Image analysis prompt**")
        st.code(workflow_config.image_analysis.prompt)
        st.markdown("**Text analysis prompt**")
        st.code(workflow_config.text_analysis.prompt)
    close_section_card()
    submitted = st.button("Run AI workflow", use_container_width=True)

    st.caption(
        "If an Ollama model is missing locally, the app will pull it automatically. "
        "To keep the workflow more laptop-friendly, the image sent to the models uses a reduced size."
    )
    st.caption(f"Config source: {workflow_config.source_path}")
    st.caption(f"Ollama endpoint: {get_ollama_base_url()}")
    if available_models:
        st.caption(f"Installed Ollama models: {', '.join(available_models)}")

    current_cache_key = workflow_cache_key(latitude, longitude, zoom, workflow_config)
    previous_result = st.session_state.get("ai_workflow_result")
    if previous_result and previous_result.get("cache_key") != current_cache_key:
        st.session_state.pop("ai_workflow_result", None)
        previous_result = None

    if submitted:
        try:
            st.session_state.pop("ai_workflow_result", None)
            with st.spinner("Downloading imagery and running the local AI workflow..."):
                st.session_state["ai_workflow_result"] = execute_governed_ai_workflow(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=zoom,
                    location_label=location_label,
                    images_dir=IMAGES_DIR,
                    config_path=MODELS_CONFIG_PATH,
                )
        except OllamaError as exc:
            st.error(str(exc))
            st.info(
                "If this is a professor/demo laptop, try a lighter Ollama model or close other apps to free memory. "
                "The server is reachable, so this error is coming from the model runtime itself."
            )
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")

    result = st.session_state.get("ai_workflow_result")
    if not result:
        if previous_result is None:
            st.info("Choose a location, adjust the settings if needed, and run the workflow to generate a fresh image analysis and risk assessment.")
        return

    image_result = result["image_result"]
    description = result["description"]
    assessment = result["assessment"]
    if result.get("cached"):
        st.info("Loaded a cached result from `database/images.csv` because the coordinates and governed settings matched a previous run.")
    else:
        st.success("Ran a fresh workflow and appended the result to `database/images.csv`.")

    st.markdown("### Results")
    open_section_card(
        "Analysis Snapshot",
        "A quick summary of the selected area, the generated satellite image, and the model output.",
    )
    left, right = st.columns([1.1, 1])
    with left:
        render_image_compat(image_result.image_path, caption=f"Saved to {image_result.image_path}")
        render_result_fact("Location", result["inputs"].get("location_label", "Custom coordinates"))
        render_result_fact(
            "Coordinates",
            f"{result['inputs']['latitude']:.4f}, {result['inputs']['longitude']:.4f}",
        )
        render_result_fact(
            "Zoom",
            f"{zoom_label_from_value(result['inputs']['zoom'])} ({result['inputs']['zoom']})",
        )
        render_result_fact("Cache key", str(result.get("cache_key", "n/a"))[:12])
        render_result_fact("Run ID", result.get("run_id", "n/a"))
        render_result_fact("Generated At (UTC)", image_result.generated_at_utc)
        with st.expander("Image metadata"):
            st.caption(f"ESRI request bbox: {', '.join(f'{value:.4f}' for value in image_result.bbox)}")
            st.caption(f"Source URL: {image_result.image_url}")

    with right:
        st.markdown("#### Image Description")
        st.write(description)
    close_section_card()

    st.markdown("### Risk Assessment")
    render_risk_badge(assessment.risk_level, assessment.flagged, assessment.risk_score)
    open_section_card(
        "Assessment Summary",
        "This section highlights the model's risk judgment and the main reasons behind it.",
    )
    st.write(assessment.summary)
    close_section_card()

    evidence_col, questions_col = st.columns(2)
    with evidence_col:
        open_section_card(
            "Supporting Evidence",
            "These are the visual clues the model used to justify the risk assessment.",
        )
        for item in assessment.evidence:
            st.write(f"- {item}")
        close_section_card()

    with questions_col:
        open_section_card(
            "Follow-Up Questions",
            "These are useful prompts for deeper analysis after the initial screening.",
        )
        for item in assessment.follow_up_questions:
            st.write(f"- {item}")
        close_section_card()

    with st.expander("Raw model output"):
        st.code(assessment.raw_response, language="json")


render_header()

with st.spinner("Loading world map and datasets..."):
    world_data, dataset_map = load_project()

st.sidebar.markdown("## Controls")
page = st.sidebar.radio("Page", options=["Environmental dashboard", "AI workflow"])

if page == "Environmental dashboard":
    render_map_page(world_data, dataset_map)
else:
    render_ai_page()

    ## To run the page do py -m streamlit run app\streamlit_app.py
