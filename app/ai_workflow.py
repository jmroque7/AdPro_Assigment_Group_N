from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
ESRI_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
DEFAULT_MODELS_CONFIG_PATH = Path("models.yaml")
DEFAULT_DATABASE_CSV_PATH = Path("database") / "images.csv"

IMAGES_CSV_FIELDS = [
    "run_id",
    "run_timestamp_utc",
    "latitude",
    "longitude",
    "zoom",
    "location_label",
    "cache_key",
    "settings_fingerprint",
    "image_path",
    "image_url",
    "image_generated_at_utc",
    "image_bbox",
    "image_prompt",
    "image_model",
    "image_settings",
    "image_description",
    "text_prompt",
    "text_model",
    "text_settings",
    "text_summary",
    "risk_level",
    "risk_score",
    "danger",
    "flagged",
    "evidence",
    "follow_up_questions",
    "raw_response",
]

class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot satisfy a request."""


class WorkflowConfigError(RuntimeError):
    """Raised when the models.yaml file is missing or invalid."""


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


@dataclass(frozen=True)
class WorkflowStageConfig:
    model: str
    prompt: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowConfig:
    image_analysis: WorkflowStageConfig
    text_analysis: WorkflowStageConfig
    source_path: Path


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


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise WorkflowConfigError(
            "PyYAML is required to read models.yaml. Install the project dependencies first."
        ) from exc
    return yaml


def _require_mapping(value: Any, section_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowConfigError(f"Section '{section_name}' must be a mapping in models.yaml.")
    return value


def _normalize_stage_config(section_name: str, raw_stage: Any) -> WorkflowStageConfig:
    stage = _require_mapping(raw_stage, section_name)
    model = str(stage.get("model", "")).strip()
    prompt = str(stage.get("prompt", "")).strip()
    settings = stage.get("settings", {})
    if not model:
        raise WorkflowConfigError(f"Section '{section_name}' is missing a non-empty 'model'.")
    if not prompt:
        raise WorkflowConfigError(f"Section '{section_name}' is missing a non-empty 'prompt'.")
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise WorkflowConfigError(f"Section '{section_name}.settings' must be a mapping.")
    return WorkflowStageConfig(model=model, prompt=prompt, settings=dict(settings))


def load_workflow_config(config_path: Path | str = DEFAULT_MODELS_CONFIG_PATH) -> WorkflowConfig:
    path = Path(config_path)
    if not path.exists():
        raise WorkflowConfigError(
            f"Workflow configuration file not found: {path}. Create models.yaml in the project root."
        )

    yaml = _load_yaml_module()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise WorkflowConfigError(f"Invalid YAML in {path}: {exc}") from exc

    root = _require_mapping(payload, "root")
    image_analysis = _normalize_stage_config("image_analysis", root.get("image_analysis"))
    text_analysis = _normalize_stage_config("text_analysis", root.get("text_analysis"))
    return WorkflowConfig(
        image_analysis=image_analysis,
        text_analysis=text_analysis,
        source_path=path,
    )


def workflow_config_as_dict(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "image_analysis": {
            "model": config.image_analysis.model,
            "prompt": config.image_analysis.prompt,
            "settings": config.image_analysis.settings,
        },
        "text_analysis": {
            "model": config.text_analysis.model,
            "prompt": config.text_analysis.prompt,
            "settings": config.text_analysis.settings,
        },
    }


def settings_fingerprint(config: WorkflowConfig) -> str:
    serialized = json.dumps(workflow_config_as_dict(config), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def workflow_cache_key(
    latitude: float,
    longitude: float,
    zoom: int,
    config: WorkflowConfig,
) -> str:
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
        "workflow": workflow_config_as_dict(config),
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ensure_images_database(csv_path: Path | str = DEFAULT_DATABASE_CSV_PATH) -> Path:
    database_path = Path(csv_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if not database_path.exists():
        with database_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=IMAGES_CSV_FIELDS)
            writer.writeheader()
    return database_path


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_loads(value: str, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _render_prompt(template: str, **values: str) -> str:
    try:
        return template.format(**values).strip()
    except KeyError as exc:
        raise WorkflowConfigError(
            f"Prompt template is missing a supported placeholder value: {exc}"
        ) from exc


def _serialize_csv_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    # Keep one physical CSV record per append by escaping embedded newlines.
    return value.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


def _deserialize_csv_cell(value: str) -> str:
    return value.replace("\\n", "\n")


def append_image_record(
    row: Mapping[str, Any],
    csv_path: Path | str = DEFAULT_DATABASE_CSV_PATH,
) -> Path:
    database_path = Path(csv_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = database_path.exists()
    serialized_row = {
        field: _serialize_csv_cell(row.get(field, ""))
        for field in IMAGES_CSV_FIELDS
    }
    frame = pd.DataFrame([serialized_row], columns=IMAGES_CSV_FIELDS)
    frame.to_csv(
        database_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    print(f"Appended 1 row to {database_path}")
    return database_path


def _ollama_options_from_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    if not settings:
        return {"temperature": 0.2}
    return {
        key: value
        for key, value in dict(settings).items()
        if key not in {"image_size"}
    } or {"temperature": 0.2}


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
    options: Mapping[str, Any] | None = None,
) -> str:
    resolved_base_url = base_url or get_ollama_base_url()
    ensure_ollama_model(model_name, base_url=resolved_base_url)
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": dict(options or {"temperature": 0.2}),
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
    prompt_template: str,
    generation_options: Mapping[str, Any] | None = None,
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

    prompt = _render_prompt(prompt_template, context_suffix=context_suffix)
    return _generate(
        model_name,
        prompt,
        images=[image_b64],
        base_url=base_url,
        timeout=1200,
        options=_ollama_options_from_settings(generation_options),
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


def _fallback_questions_from_text(text: str) -> list[str]:
    questions = [
        "Is the same pattern visible in imagery from prior months or years?",
        "Are nearby rivers, wetlands, or vegetation corridors showing signs of stress?",
        "Would higher-resolution imagery confirm whether the visible disturbance is temporary or persistent?",
    ]
    extracted = re.findall(r"([^.!?]*\?)", text)
    cleaned = [item.strip() for item in extracted if item.strip()]
    return (cleaned[:3] + questions)[:3]


def fallback_risk_response(raw_response: str, image_description: str) -> dict[str, Any]:
    cleaned_response = " ".join(raw_response.split()).strip()
    summary = cleaned_response[:280] if cleaned_response else image_description[:280]
    if not summary:
        summary = "The model returned an unstructured response, so this assessment was reconstructed from the image description."

    evidence: list[str] = []
    for source_text in (cleaned_response, image_description):
        parts = re.split(r"(?<=[.!?])\s+|\n+", source_text)
        for part in parts:
            candidate = part.strip(" -\t")
            if len(candidate) >= 24 and candidate not in evidence:
                evidence.append(candidate[:160])
            if len(evidence) >= 4:
                break
        if len(evidence) >= 4:
            break

    if not evidence:
        evidence = [
            "The image description was available but the model response was not valid JSON.",
            image_description[:160] if image_description else "No structured evidence was returned.",
        ]

    heuristic_score = _infer_score_from_text(image_description, summary, evidence)
    if heuristic_score < 35:
        risk_level = "low"
    elif heuristic_score < 70:
        risk_level = "medium"
    else:
        risk_level = "high"

    severe_hits = _keyword_score(" ".join(evidence + [summary, image_description]), SEVERE_RISK_TERMS)
    flagged = heuristic_score >= 55 or severe_hits >= 18

    return {
        "flagged": flagged,
        "risk_level": risk_level,
        "risk_score": heuristic_score,
        "summary": summary,
        "evidence": evidence[:4],
        "follow_up_questions": _fallback_questions_from_text(raw_response or image_description),
    }


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
    prompt_template: str,
    generation_options: Mapping[str, Any] | None = None,
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

    prompt = _render_prompt(
        prompt_template,
        image_description=image_description,
        location_context=location_context,
    )

    raw_response = _generate(
        model_name,
        prompt,
        base_url=base_url,
        timeout=600,
        options=_ollama_options_from_settings(generation_options),
    )
    try:
        parsed = extract_json_object(raw_response)
    except (ValueError, json.JSONDecodeError):
        parsed = fallback_risk_response(raw_response, image_description)
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


def _image_result_from_row(row: Mapping[str, str]) -> ImageDownloadResult:
    return ImageDownloadResult(
        image_path=Path(row["image_path"]),
        image_url=_deserialize_csv_cell(row["image_url"]),
        bbox=tuple(_json_loads(row["image_bbox"], [])),
        generated_at_utc=row["image_generated_at_utc"],
    )


def _assessment_from_row(row: Mapping[str, str]) -> RiskAssessment:
    return RiskAssessment(
        flagged=str(row.get("flagged", "")).strip().lower() == "true",
        risk_level=_deserialize_csv_cell(row["risk_level"]),
        risk_score=int(row["risk_score"]),
        summary=_deserialize_csv_cell(row["text_summary"]),
        evidence=list(_json_loads(row["evidence"], [])),
        follow_up_questions=list(_json_loads(row["follow_up_questions"], [])),
        raw_response=_deserialize_csv_cell(row["raw_response"]),
    )


def result_from_database_row(row: Mapping[str, str]) -> dict[str, object]:
    image_result = _image_result_from_row(row)
    assessment = _assessment_from_row(row)
    return {
        "run_id": row["run_id"],
        "inputs": {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "zoom": int(row["zoom"]),
            "vision_model": _deserialize_csv_cell(row["image_model"]),
            "risk_model": _deserialize_csv_cell(row["text_model"]),
            "location_label": _deserialize_csv_cell(row["location_label"]),
        },
        "image_result": image_result,
        "description": _deserialize_csv_cell(row["image_description"]),
        "assessment": assessment,
        "cache_key": row["cache_key"],
        "settings_fingerprint": row["settings_fingerprint"],
        "cached": True,
        "workflow_config": {
            "image_analysis": {
                "model": _deserialize_csv_cell(row["image_model"]),
                "prompt": _deserialize_csv_cell(row["image_prompt"]),
                "settings": _json_loads(row["image_settings"], {}),
            },
            "text_analysis": {
                "model": _deserialize_csv_cell(row["text_model"]),
                "prompt": _deserialize_csv_cell(row["text_prompt"]),
                "settings": _json_loads(row["text_settings"], {}),
            },
        },
    }


def find_cached_workflow_result(
    cache_key: str,
    csv_path: Path | str = DEFAULT_DATABASE_CSV_PATH,
) -> dict[str, object] | None:
    database_path = Path(csv_path)
    if not database_path.exists():
        return None

    with database_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reversed(list(reader)):
            if row.get("cache_key") != cache_key:
                continue
            image_path = Path(row.get("image_path", ""))
            if not image_path.exists():
                continue
            return result_from_database_row(row)
    return None


def append_workflow_record(
    result: Mapping[str, object],
    config: WorkflowConfig,
    cache_key: str,
    csv_path: Path | str = DEFAULT_DATABASE_CSV_PATH,
) -> Path:
    database_path = ensure_images_database(csv_path)
    image_result = result["image_result"]
    assessment = result["assessment"]
    inputs = result["inputs"]
    if not isinstance(image_result, ImageDownloadResult):
        raise TypeError("result['image_result'] must be an ImageDownloadResult")
    if not isinstance(assessment, RiskAssessment):
        raise TypeError("result['assessment'] must be a RiskAssessment")

    row = {
        "run_id": str(result["run_id"]),
        "run_timestamp_utc": str(result.get("run_timestamp_utc") or image_result.generated_at_utc),
        "latitude": f"{float(inputs['latitude']):.6f}",
        "longitude": f"{float(inputs['longitude']):.6f}",
        "zoom": str(int(inputs["zoom"])),
        "location_label": str(inputs["location_label"]),
        "cache_key": cache_key,
        "settings_fingerprint": settings_fingerprint(config),
        "image_path": str(image_result.image_path),
        "image_url": image_result.image_url,
        "image_generated_at_utc": image_result.generated_at_utc,
        "image_bbox": _json_dumps(list(image_result.bbox)),
        "image_prompt": config.image_analysis.prompt,
        "image_model": config.image_analysis.model,
        "image_settings": _json_dumps(config.image_analysis.settings),
        "image_description": str(result["description"]),
        "text_prompt": config.text_analysis.prompt,
        "text_model": config.text_analysis.model,
        "text_settings": _json_dumps(config.text_analysis.settings),
        "text_summary": assessment.summary,
        "risk_level": assessment.risk_level,
        "risk_score": str(assessment.risk_score),
        "danger": "Y" if assessment.flagged else "N",
        "flagged": str(assessment.flagged),
        "evidence": _json_dumps(assessment.evidence),
        "follow_up_questions": _json_dumps(assessment.follow_up_questions),
        "raw_response": assessment.raw_response,
    }

    return append_image_record(row, csv_path=database_path)


def execute_governed_ai_workflow(
    latitude: float,
    longitude: float,
    zoom: int,
    *,
    location_label: str,
    images_dir: Path,
    config_path: Path | str = DEFAULT_MODELS_CONFIG_PATH,
    database_csv_path: Path | str = DEFAULT_DATABASE_CSV_PATH,
) -> dict[str, object]:
    config = load_workflow_config(config_path)
    cache_key = workflow_cache_key(latitude, longitude, zoom, config)
    cached_result = find_cached_workflow_result(cache_key, csv_path=database_csv_path)
    if cached_result is not None:
        return cached_result

    check_ollama_available()
    image_size = int(config.image_analysis.settings.get("image_size", 512))
    image_result = download_esri_world_imagery(
        latitude,
        longitude,
        zoom,
        images_dir,
        image_size=image_size,
    )
    description = describe_image_with_ollama(
        image_result.image_path,
        model_name=config.image_analysis.model,
        prompt_template=config.image_analysis.prompt,
        generation_options=config.image_analysis.settings,
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        bbox=image_result.bbox,
    )
    assessment = assess_environmental_risk(
        description,
        model_name=config.text_analysis.model,
        prompt_template=config.text_analysis.prompt,
        generation_options=config.text_analysis.settings,
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        bbox=image_result.bbox,
    )
    result = {
        "run_id": hashlib.sha256(
            f"{cache_key}:{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
        ).hexdigest()[:32],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "latitude": latitude,
            "longitude": longitude,
            "zoom": zoom,
            "vision_model": config.image_analysis.model,
            "risk_model": config.text_analysis.model,
            "location_label": location_label,
        },
        "image_result": image_result,
        "description": description,
        "assessment": assessment,
        "cache_key": cache_key,
        "settings_fingerprint": settings_fingerprint(config),
        "cached": False,
        "workflow_config": workflow_config_as_dict(config),
    }
    append_workflow_record(result, config, cache_key, csv_path=database_csv_path)
    return result
