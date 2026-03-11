from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
ESRI_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot satisfy a request."""


@dataclass(frozen=True)
class ImageDownloadResult:
    image_path: Path
    image_url: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class RiskAssessment:
    flagged: bool
    risk_level: str
    risk_score: int
    summary: str
    evidence: list[str]
    follow_up_questions: list[str]
    raw_response: str


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "location"


def build_bbox(latitude: float, longitude: float, zoom: int, image_size: int = 768) -> tuple[float, float, float, float]:
    """
    Approximate a 4326 bounding box around a point using a slippy-map zoom level.
    """
    latitude = max(min(latitude, 85.0), -85.0)
    longitude = ((longitude + 180.0) % 360.0) - 180.0

    meters_per_pixel = (
        156543.03392 * math.cos(math.radians(latitude)) / (2 ** max(0, zoom))
    )
    half_width_m = meters_per_pixel * image_size / 2
    half_height_m = meters_per_pixel * image_size / 2

    lon_deg_per_meter = 1 / max(111320 * math.cos(math.radians(latitude)), 1e-6)
    lat_deg_per_meter = 1 / 110574

    lon_delta = half_width_m * lon_deg_per_meter
    lat_delta = half_height_m * lat_deg_per_meter

    min_lon = max(-180.0, longitude - lon_delta)
    max_lon = min(180.0, longitude + lon_delta)
    min_lat = max(-85.0, latitude - lat_delta)
    max_lat = min(85.0, latitude + lat_delta)
    return (min_lon, min_lat, max_lon, max_lat)


def download_esri_world_imagery(
    latitude: float,
    longitude: float,
    zoom: int,
    output_dir: Path,
    *,
    image_size: int = 768,
    session: requests.sessions.Session | None = None,
) -> ImageDownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = build_bbox(latitude, longitude, zoom, image_size=image_size)
    bbox_str = ",".join(f"{value:.6f}" for value in bbox)

    params = {
        "bbox": bbox_str,
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{image_size},{image_size}",
        "format": "jpg",
        "f": "image",
    }

    client = session or requests
    response = client.get(ESRI_EXPORT_URL, params=params, timeout=90)
    response.raise_for_status()

    filename = (
        f"esri_{sanitize_filename_part(f'{latitude:.4f}_{longitude:.4f}_z{zoom}')}.jpg"
    )
    image_path = output_dir / filename
    image_path.write_bytes(response.content)

    image_url = f"{ESRI_EXPORT_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    return ImageDownloadResult(image_path=image_path, image_url=image_url, bbox=bbox)


def check_ollama_available(base_url: str = OLLAMA_BASE_URL) -> None:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            "Ollama is not reachable on http://127.0.0.1:11434. Start the Ollama app or service first."
        ) from exc


def _ollama_tags(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    check_ollama_available(base_url=base_url)
    response = requests.get(f"{base_url}/api/tags", timeout=20)
    response.raise_for_status()
    payload = response.json()
    return [model["name"] for model in payload.get("models", [])]


def ensure_ollama_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> None:
    available_models = _ollama_tags(base_url=base_url)
    if model_name in available_models:
        return

    response = requests.post(
        f"{base_url}/api/pull",
        json={"name": model_name, "stream": False},
        timeout=1800,
    )
    response.raise_for_status()


def _generate(
    model_name: str,
    prompt: str,
    *,
    images: list[str] | None = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 600,
) -> str:
    ensure_ollama_model(model_name, base_url=base_url)
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if images:
        payload["images"] = images

    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def describe_image_with_ollama(
    image_path: Path,
    *,
    model_name: str = "llava:7b",
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = (
        "You are analyzing a satellite image for an environmental monitoring tool. "
        "Describe only what is visible. Mention land cover, vegetation density, water, "
        "roads, settlements, bare soil, burn scars, mining traces, deforestation patterns, "
        "flooding or drought cues if visible. Avoid guessing the exact location."
    )
    return _generate(
        model_name,
        prompt,
        images=[image_b64],
        base_url=base_url,
        timeout=1200,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start : end + 1])


def assess_environmental_risk(
    image_description: str,
    *,
    model_name: str = "llama3.2:3b",
    base_url: str = OLLAMA_BASE_URL,
) -> RiskAssessment:
    prompt = f"""
You are assisting an environmental risk triage workflow.

Given the satellite-image description below, ask yourself the right hidden diagnostic questions and then return only a JSON object.

Description:
\"\"\"
{image_description}
\"\"\"

Return JSON with exactly these keys:
- flagged: boolean
- risk_level: one of ["low", "medium", "high"]
- risk_score: integer from 0 to 100
- summary: short string
- evidence: array with 3 to 5 short bullet-style strings
- follow_up_questions: array with 3 short questions the analyst should investigate next

Flag the area when the description suggests probable deforestation, land degradation, drought stress, flooding damage, mining scars, wildfire scars, habitat fragmentation, rapid urban encroachment, or other visible environmental stressors.
If the evidence is weak, keep the score lower and explain the uncertainty in the summary.
Return JSON only.
""".strip()

    raw_response = _generate(model_name, prompt, base_url=base_url, timeout=600)
    parsed = extract_json_object(raw_response)

    flagged = bool(parsed.get("flagged", False))
    risk_level = str(parsed.get("risk_level", "low")).lower()
    risk_score = max(0, min(100, int(parsed.get("risk_score", 0))))
    summary = str(parsed.get("summary", "")).strip()
    evidence = [str(item).strip() for item in parsed.get("evidence", []) if str(item).strip()]
    follow_up_questions = [
        str(item).strip()
        for item in parsed.get("follow_up_questions", [])
        if str(item).strip()
    ]

    return RiskAssessment(
        flagged=flagged,
        risk_level=risk_level,
        risk_score=risk_score,
        summary=summary,
        evidence=evidence,
        follow_up_questions=follow_up_questions,
        raw_response=raw_response,
    )
