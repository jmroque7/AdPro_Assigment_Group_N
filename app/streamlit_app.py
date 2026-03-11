from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from app.ai_workflow import (
    OllamaError,
    assess_environmental_risk,
    check_ollama_available,
    describe_image_with_ollama,
    download_esri_world_imagery,
)
from app.okavango import OkavangoConfig, OkavangoProject


st.set_page_config(
    page_title="Project Okavango",
    page_icon="map",
    layout="wide",
)

PINK_CMAP = "RdPu"
IMAGES_DIR = Path("images")


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


@st.cache_data(show_spinner=False)
def load_project() -> Tuple[gpd.GeoDataFrame, Dict[str, pd.DataFrame]]:
    cfg = OkavangoConfig(project_root=Path("."), download=True, latest_year_only=False)
    project = OkavangoProject(cfg)
    return project.world, project.datasets


def render_header() -> None:
    st.markdown(
        """
        <style>
          .ok-title { color: #ff4fa3; font-weight: 800; font-size: 34px; }
          .ok-sub { color: #8c2f6b; font-size: 16px; }
          .risk-card {
              border-radius: 16px;
              padding: 1rem 1.2rem;
              color: white;
              font-weight: 700;
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

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("### World map")
        fig, ax = plt.subplots(figsize=(13, 7))
        merged.plot(
            column=metric_col,
            cmap=PINK_CMAP,
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )
        ax.set_axis_off()
        ax.set_title(f"{dataset_choice} - {year_choice}", fontsize=16)

        cbar_ax = fig.axes[-1]
        cbar_ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

        if show_labels:
            centroids = merged.copy()
            centroids["centroid"] = centroids.geometry.centroid
            for _, row in centroids.dropna(subset=[metric_col]).head(40).iterrows():
                ax.text(
                    row["centroid"].x,
                    row["centroid"].y,
                    str(row.get("NAME", ""))[:10],
                    fontsize=6,
                    alpha=0.5,
                )
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown("### Key metrics")
        c1, c2 = st.columns(2)
        c1.metric("Selected year", f"{year_choice}")
        c2.metric("Latest available", f"{latest_year}")
        c3, c4 = st.columns(2)
        c3.metric("Countries with data", f"{n_with_data}/{n_total_countries}")
        c4.metric("Missing countries", f"{missing_pct}%")
        st.write(
            "Grey countries have no matched data for the selected year. "
            "Use the year selector to compare the latest available data with prior years."
        )

    st.markdown("### Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Top 5 countries")
        st.dataframe(top5.rename(columns={metric_col: "value"}), use_container_width=True)
        fig_top, ax_top = plt.subplots(figsize=(7, 4))
        ax_top.barh(top5["Entity"], top5[metric_col])
        ax_top.invert_yaxis()
        ax_top.set_xlabel("Value")
        st.pyplot(fig_top, use_container_width=True)

    with col_b:
        st.markdown("#### Bottom 5 countries")
        st.dataframe(bottom5.rename(columns={metric_col: "value"}), use_container_width=True)
        fig_bottom, ax_bottom = plt.subplots(figsize=(7, 4))
        ax_bottom.barh(bottom5["Entity"], bottom5[metric_col])
        ax_bottom.invert_yaxis()
        ax_bottom.set_xlabel("Value")
        st.pyplot(fig_bottom, use_container_width=True)

    st.markdown("### Distribution")
    values = df_year[metric_col].dropna()
    fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
    ax_hist.hist(values, bins=30, alpha=0.75)
    ax_hist.set_title("Histogram of values across countries")
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Number of countries")
    st.pyplot(fig_hist, use_container_width=True)


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


def run_ai_workflow(
    latitude: float,
    longitude: float,
    zoom: int,
    vision_model: str,
    risk_model: str,
) -> dict[str, object]:
    check_ollama_available()
    image_result = download_esri_world_imagery(latitude, longitude, zoom, IMAGES_DIR)
    description = describe_image_with_ollama(image_result.image_path, model_name=vision_model)
    assessment = assess_environmental_risk(description, model_name=risk_model)
    return {
        "inputs": {
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "vision_model": vision_model,
            "risk_model": risk_model,
        },
        "image_result": image_result,
        "description": description,
        "assessment": assessment,
    }


def render_ai_page() -> None:
    st.subheader("AI Workflow: Image-Based Environmental Risk Check")
    st.write(
        "Select coordinates, download recent ESRI World Imagery for that area, generate a satellite-image description with a local vision model, and then run a second model to flag environmental risk."
    )

    with st.form("ai_workflow_form"):
        col1, col2, col3 = st.columns(3)
        latitude = col1.number_input("Latitude", min_value=-85.0, max_value=85.0, value=-19.0, step=0.1)
        longitude = col2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=23.0, step=0.1)
        zoom = col3.slider("Zoom", min_value=3, max_value=17, value=10)

        col4, col5 = st.columns(2)
        vision_model = col4.text_input("Vision model", value="llava:7b")
        risk_model = col5.text_input("Risk model", value="llama3.2:3b")
        submitted = st.form_submit_button("Run AI workflow", use_container_width=True)

    st.caption(
        "The app will pull the Ollama models automatically if they are missing locally. "
        "Large models may take time on the first run."
    )

    current_inputs = {
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
        "vision_model": vision_model,
        "risk_model": risk_model,
    }
    previous_result = st.session_state.get("ai_workflow_result")
    if previous_result and previous_result.get("inputs") != current_inputs:
        st.session_state.pop("ai_workflow_result", None)
        previous_result = None

    if submitted:
        try:
            with st.spinner("Downloading imagery and running the local AI workflow..."):
                st.session_state["ai_workflow_result"] = run_ai_workflow(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=zoom,
                    vision_model=vision_model,
                    risk_model=risk_model,
                )
        except OllamaError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Workflow failed: {exc}")

    result = st.session_state.get("ai_workflow_result")
    if not result:
        if previous_result is None:
            st.info("Change coordinates, zoom, or model names and run the workflow again to generate a fresh image and risk assessment.")
        return

    image_result = result["image_result"]
    description = result["description"]
    assessment = result["assessment"]

    st.markdown("### Results")
    left, right = st.columns([1.1, 1])
    with left:
        st.image(str(image_result.image_path), caption=f"Saved to {image_result.image_path}", use_container_width=True)
        st.caption(f"ESRI request bbox: {', '.join(f'{value:.4f}' for value in image_result.bbox)}")
        st.caption(f"Source URL: {image_result.image_url}")

    with right:
        st.markdown("#### Vision model description")
        st.write(description)

    st.markdown("### Risk assessment")
    render_risk_badge(assessment.risk_level, assessment.flagged, assessment.risk_score)
    st.write(assessment.summary)

    evidence_col, questions_col = st.columns(2)
    with evidence_col:
        st.markdown("#### Evidence used by the model")
        for item in assessment.evidence:
            st.write(f"- {item}")

    with questions_col:
        st.markdown("#### Follow-up questions")
        for item in assessment.follow_up_questions:
            st.write(f"- {item}")

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
