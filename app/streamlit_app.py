from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

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
    get_ollama_base_url,
    list_ollama_models,
)
from app.okavango import OkavangoConfig, OkavangoProject


st.set_page_config(
    page_title="Project Okavango",
    page_icon="map",
    layout="wide",
)

PINK_CMAP = "RdPu"
IMAGES_DIR = Path("images")
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
          :root {
              --ok-card-border: rgba(255, 255, 255, 0.09);
              --ok-card-bg: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
              --ok-muted: #b8bcc8;
              --ok-accent: #ff4fa3;
              --ok-accent-soft: rgba(255, 79, 163, 0.14);
          }
          .ok-title { color: #ff4fa3; font-weight: 800; font-size: 34px; }
          .ok-sub { color: #8c2f6b; font-size: 16px; }
          .ok-section-card {
              border: 1px solid var(--ok-card-border);
              border-radius: 18px;
              padding: 1rem 1rem 0.45rem 1rem;
              margin-bottom: 1rem;
              background: var(--ok-card-bg);
              box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
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


def run_ai_workflow(
    latitude: float,
    longitude: float,
    zoom: int,
    vision_model: str,
    risk_model: str,
    location_label: str,
) -> dict[str, object]:
    check_ollama_available()
    image_result = download_esri_world_imagery(latitude, longitude, zoom, IMAGES_DIR, image_size=512)
    description = describe_image_with_ollama(
        image_result.image_path,
        model_name=vision_model,
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        bbox=image_result.bbox,
    )
    assessment = assess_environmental_risk(
        description,
        model_name=risk_model,
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        bbox=image_result.bbox,
    )
    return {
        "run_id": uuid4().hex,
        "inputs": {
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "vision_model": vision_model,
            "risk_model": risk_model,
            "location_label": location_label,
        },
        "image_result": image_result,
        "description": description,
        "assessment": assessment,
    }


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
        "Choose the satellite view scale and the local Ollama models that will describe the image and assess risk.",
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
    vision_model = col4.text_input("Vision model", value=st.session_state.get("ai_vision_model", "llava:7b"), key="ai_vision_model")
    risk_model = col5.text_input("Risk model", value=st.session_state.get("ai_risk_model", "llama3.2:3b"), key="ai_risk_model")
    close_section_card()
    submitted = st.button("Run AI workflow", use_container_width=True)

    st.caption(
        "If an Ollama model is missing locally, the app will pull it automatically. "
        "To keep the workflow more laptop-friendly, the image sent to the models uses a reduced size."
    )
    st.caption(f"Ollama endpoint: {get_ollama_base_url()}")
    if available_models:
        st.caption(f"Installed Ollama models: {', '.join(available_models)}")

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
            st.session_state.pop("ai_workflow_result", None)
            with st.spinner("Downloading imagery and running the local AI workflow..."):
                st.session_state["ai_workflow_result"] = run_ai_workflow(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=zoom,
                    vision_model=vision_model,
                    risk_model=risk_model,
                    location_label=location_label,
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
