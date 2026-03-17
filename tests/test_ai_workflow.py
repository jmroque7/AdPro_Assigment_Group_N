from __future__ import annotations

import pytest
import requests

from app.ai_workflow import (
    OllamaError,
    _read_ollama_error,
    _raise_for_ollama_response,
    _normalized_risk_outputs,
    build_bbox,
    check_ollama_available,
    extract_json_object,
    get_ollama_base_url,
)


def test_build_bbox_returns_valid_bounds() -> None:
    min_lon, min_lat, max_lon, max_lat = build_bbox(38.7223, -9.1393, zoom=10, image_size=512)

    assert min_lon < max_lon
    assert min_lat < max_lat
    assert -180 <= min_lon <= 180
    assert -85 <= min_lat <= 85
    assert -180 <= max_lon <= 180
    assert -85 <= max_lat <= 85


def test_extract_json_object_supports_fenced_json() -> None:
    payload = """
    Here is the result:
    ```json
    {"flagged": true, "risk_level": "high", "risk_score": 88}
    ```
    """

    parsed = extract_json_object(payload)

    assert parsed["flagged"] is True
    assert parsed["risk_level"] == "high"
    assert parsed["risk_score"] == 88


def test_get_ollama_base_url_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert get_ollama_base_url() == "http://localhost:11434"


def test_get_ollama_base_url_uses_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.internal:8080/")

    assert get_ollama_base_url() == "https://ollama.internal:8080"


def test_get_ollama_base_url_ignores_placeholder_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://<your-host>:11434")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert get_ollama_base_url() == "http://127.0.0.1:11434"


def test_check_ollama_available_reports_resolved_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(OllamaError) as exc_info:
        check_ollama_available(base_url="http://localhost:11434")

    assert "http://localhost:11434" in str(exc_info.value)


def test_read_ollama_error_prefers_json_error_message() -> None:
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"model requires more system memory"}'
    response.headers["Content-Type"] = "application/json"

    assert _read_ollama_error(response) == "model requires more system memory"


def test_raise_for_ollama_response_wraps_http_error() -> None:
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"something went wrong"}'
    response.url = "http://127.0.0.1:11434/api/generate"
    response.headers["Content-Type"] = "application/json"

    with pytest.raises(OllamaError) as exc_info:
        _raise_for_ollama_response(response, action="generate a response", model_name="llava:7b")

    assert "llava:7b" in str(exc_info.value)
    assert "something went wrong" in str(exc_info.value)


def test_normalized_risk_outputs_uses_more_than_anchor_scores() -> None:
    parsed = {
        "flagged": True,
        "risk_level": "high",
        "risk_score": 80,
        "summary": "Visible deforestation and mining scars suggest strong environmental stress.",
        "evidence": [
            "Patchy forest clearing is visible.",
            "Mining-related bare soil appears in multiple areas.",
            "Road access fragments surrounding vegetation.",
        ],
        "follow_up_questions": ["How recent is the land clearing?", "Is mining activity expanding?", "Are nearby water bodies affected?"],
    }

    flagged, risk_level, risk_score, *_ = _normalized_risk_outputs(
        parsed=parsed,
        image_description="Satellite view shows deforestation, mining, bare soil and fragmented vegetation.",
    )

    assert flagged is True
    assert risk_level == "high"
    assert risk_score != 80
    assert risk_score > 70


def test_normalized_risk_outputs_reduces_uncertain_low_signal_cases() -> None:
    parsed = {
        "flagged": True,
        "risk_level": "medium",
        "risk_score": 40,
        "summary": "Possible disturbance is visible, but evidence is limited and uncertain.",
        "evidence": [
            "Mostly natural vegetation is still present.",
            "No clear signs of major damage are visible.",
            "Possible bare soil patches need confirmation.",
        ],
        "follow_up_questions": ["Is there seasonality in vegetation cover?", "Could soil exposure be temporary?", "Are there higher-resolution images available?"],
    }

    flagged, risk_level, risk_score, *_ = _normalized_risk_outputs(
        parsed=parsed,
        image_description="Mostly intact vegetation with limited disturbance and uncertain small bare soil patches.",
    )

    assert flagged is False
    assert risk_level == "low"
    assert risk_score < 40
