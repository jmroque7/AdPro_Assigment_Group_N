from __future__ import annotations

import base64
import json
import math
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    generated_at_utc: str


@dataclass(frozen=True)
class RiskAssessment:
    flagged: bool
    risk_level: str
    risk_score: int
    summary: str
    evidence: list[str]
    follow_up_questions: list[str]
    raw_response: str


SEVERE_RISK_TERMS = {
    "deforestation": 12,
    "clear-cut": 12,
    "clearcut": 12,
    "burn scar": 10,
    "burn scars": 10,
    "wildfire": 10,
    "mining": 12,
    "mine": 10,
    "erosion": 8,
    "degradation": 8,
    "degraded": 8,
    "fragmentation": 9,
    "flood damage": 9,
    "flooding": 7,
    "drought": 8,
    "bare soil": 6,
    "settlement expansion": 8,
    "urban encroachment": 10,
    "habitat loss": 10,
}

LOW_RISK_TERMS = {
    "dense vegetation": -8,
    "healthy vegetation": -8,
    "intact forest": -10,
    "closed canopy": -9,
    "protected area": -6,
    "limited disturbance": -7,
    "mostly natural": -7,
    "no obvious damage": -12,
    "no clear signs": -10,
    "stable water": -4,
}

UNCERTAINTY_TERMS = {
    "uncertain": -6,
    "unclear": -6,
    "possible": -4,
    "may indicate": -4,
    "might": -3,
    "appears to": -3,
    "difficult to confirm": -7,
    "limited evidence": -8,
    "weak evidence": -8,
}


def get_ollama_base_url() -> str:
    raw_value = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or OLLAMA_BASE_URL
    ).strip()
    if not raw_value:
        return OLLAMA_BASE_URL
    if "<" in raw_value or ">" in raw_value or "your-host" in raw_value.lower():
        return OLLAMA_BASE_URL

    if "://" not in raw_value:
        raw_value = f"http://{raw_value}"

    parsed = urlparse(raw_value)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    normalized_path = path.rstrip("/")
    return f"{scheme}://{netloc}{normalized_path}"


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "location"


def _read_ollama_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    text = response.text.strip()
    if text:
        return text
    return f"HTTP {response.status_code}"


def _raise_for_ollama_response(response: requests.Response, action: str, model_name: str | None = None) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        suffix = f" for model '{model_name}'" if model_name else ""
        detail = _read_ollama_error(response)
        raise OllamaError(f"Ollama failed while trying to {action}{suffix}: {detail}") from exc


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
    image_size: int = 512,
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
    return ImageDownloadResult(
        image_path=image_path,
        image_url=image_url,
        bbox=bbox,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def check_ollama_available(base_url: str | None = None) -> None:
    resolved_base_url = base_url or get_ollama_base_url()
    try:
        response = requests.get(f"{resolved_base_url}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama is not reachable on {resolved_base_url}. "
            "Start the Ollama app or service first, or set OLLAMA_HOST/OLLAMA_BASE_URL to the correct endpoint."
        ) from exc


def _ollama_tags(base_url: str | None = None) -> list[str]:
    resolved_base_url = base_url or get_ollama_base_url()
    check_ollama_available(base_url=resolved_base_url)
    response = requests.get(f"{resolved_base_url}/api/tags", timeout=20)
    response.raise_for_status()
    payload = response.json()
    return [model["name"] for model in payload.get("models", [])]


def list_ollama_models(base_url: str | None = None) -> list[str]:
    return sorted(_ollama_tags(base_url=base_url))


def ensure_ollama_model(model_name: str, base_url: str | None = None) -> None:
    resolved_base_url = base_url or get_ollama_base_url()
    available_models = _ollama_tags(base_url=resolved_base_url)
    if model_name in available_models:
        return

    response = requests.post(
        f"{resolved_base_url}/api/pull",
        json={"name": model_name, "stream": False},
        timeout=1800,
    )
    _raise_for_ollama_response(response, action="pull the model", model_name=model_name)


def _generate(
    model_name: str,
    prompt: str,
    *,
    images: list[str] | None = None,
    base_url: str | None = None,
    timeout: int = 600,
) -> str:
    resolved_base_url = base_url or get_ollama_base_url()
    ensure_ollama_model(model_name, base_url=resolved_base_url)
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if images:
        payload["images"] = images

    response = requests.post(
        f"{resolved_base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    _raise_for_ollama_response(response, action="generate a response", model_name=model_name)
    data = response.json()
    return data.get("response", "").strip()


def describe_image_with_ollama(
    image_path: Path,
    *,
    model_name: str = "llava:7b",
    latitude: float | None = None,
    longitude: float | None = None,
    zoom: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    base_url: str | None = None,
) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    context_bits: list[str] = []
    if latitude is not None and longitude is not None:
        context_bits.append(f"center coordinates: ({latitude:.4f}, {longitude:.4f})")
    if zoom is not None:
        context_bits.append(f"zoom level: {zoom}")
    if bbox is not None:
        bbox_text = ", ".join(f"{value:.4f}" for value in bbox)
        context_bits.append(f"bounding box: [{bbox_text}]")
    context_suffix = ""
    if context_bits:
        context_suffix = " Context for scale only: " + "; ".join(context_bits) + "."

    prompt = (
        "You are analyzing a satellite image for an environmental monitoring tool. "
        "Describe only what is visible. Mention land cover, vegetation density, water, "
        "roads, settlements, bare soil, burn scars, mining traces, deforestation patterns, "
        "flooding or drought cues if visible. Avoid guessing the exact location."
        f"{context_suffix}"
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


def _keyword_score(text: str, weights: dict[str, int]) -> int:
    lowered = text.lower()
    return sum(weight for phrase, weight in weights.items() if phrase in lowered)


def _infer_score_from_text(image_description: str, summary: str, evidence: list[str]) -> int:
    combined = " ".join([image_description, summary, *evidence]).strip()
    severe_score = _keyword_score(combined, SEVERE_RISK_TERMS)
    low_risk_score = _keyword_score(combined, LOW_RISK_TERMS)
    uncertainty_penalty = _keyword_score(combined, UNCERTAINTY_TERMS)

    raw_score = 30 + severe_score + low_risk_score + uncertainty_penalty
    if len(evidence) >= 4:
        raw_score += 3
    if len(evidence) <= 1:
        raw_score -= 4
    return max(0, min(100, raw_score))


def _normalized_risk_outputs(
    *,
    parsed: dict[str, Any],
    image_description: str,
) -> tuple[bool, str, int, str, list[str], list[str]]:
    summary = str(parsed.get("summary", "")).strip()
    evidence = [str(item).strip() for item in parsed.get("evidence", []) if str(item).strip()]
    follow_up_questions = [
        str(item).strip()
        for item in parsed.get("follow_up_questions", [])
        if str(item).strip()
    ]

    model_score = max(0, min(100, int(parsed.get("risk_score", 0))))
    heuristic_score = _infer_score_from_text(image_description, summary, evidence)
    blended_score = round((0.55 * model_score) + (0.45 * heuristic_score))

    if blended_score < 35:
        risk_level = "low"
    elif blended_score < 70:
        risk_level = "medium"
    else:
        risk_level = "high"

    severe_hits = _keyword_score(" ".join(evidence + [summary, image_description]), SEVERE_RISK_TERMS)
    flagged = blended_score >= 55 or severe_hits >= 18
    return flagged, risk_level, blended_score, summary, evidence, follow_up_questions


def assess_environmental_risk(
    image_description: str,
    *,
    model_name: str = "llama3.2:3b",
    latitude: float | None = None,
    longitude: float | None = None,
    zoom: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    base_url: str | None = None,
) -> RiskAssessment:
    location_lines: list[str] = []
    if latitude is not None and longitude is not None:
        location_lines.append(f"Center coordinates: ({latitude:.4f}, {longitude:.4f})")
    if zoom is not None:
        location_lines.append(f"Zoom level: {zoom}")
    if bbox is not None:
        location_lines.append(f"Bounding box: [{', '.join(f'{value:.4f}' for value in bbox)}]")
    location_context = "\n".join(location_lines)
    if location_context:
        location_context = f"\nLocation context:\n{location_context}\n"

    prompt = f"""
You are assisting an environmental risk triage workflow.
{location_context}

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
Use the full score range instead of defaulting to round anchor values. As a guide:
- 0 to 15: little or no visible concern
- 16 to 34: minor or uncertain signals
- 35 to 54: moderate concern needing follow-up
- 55 to 74: strong visible concern
- 75 to 100: severe visible concern
Avoid always returning 40 or 80 unless the evidence truly fits those exact values.
Return JSON only.
""".strip()

    raw_response = _generate(model_name, prompt, base_url=base_url, timeout=600)
    parsed = extract_json_object(raw_response)
    flagged, risk_level, risk_score, summary, evidence, follow_up_questions = _normalized_risk_outputs(
        parsed=parsed,
        image_description=image_description,
    )

    return RiskAssessment(
        flagged=flagged,
        risk_level=risk_level,
        risk_score=risk_score,
        summary=summary,
        evidence=evidence,
        follow_up_questions=follow_up_questions,
        raw_response=raw_response,
    )
