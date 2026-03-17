from __future__ import annotations

from pathlib import Path

import pytest
import requests

from app.ai_workflow import (
    IMAGES_CSV_FIELDS,
    OllamaError,
    RiskAssessment,
    WorkflowConfig,
    WorkflowConfigError,
    WorkflowStageConfig,
    _read_ollama_error,
    _raise_for_ollama_response,
    _normalized_risk_outputs,
    append_image_record,
    append_workflow_record,
    build_bbox,
    check_ollama_available,
    ensure_images_database,
    execute_governed_ai_workflow,
    extract_json_object,
    find_cached_workflow_result,
    get_ollama_base_url,
    load_workflow_config,
    settings_fingerprint,
    workflow_cache_key,
    ImageDownloadResult,
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


def test_load_workflow_config_reads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
image_analysis:
  model: llava:7b
  prompt: "Describe the image.{context_suffix}"
  settings:
    temperature: 0.2
    image_size: 512
text_analysis:
  model: llama3.2:3b
  prompt: "Assess it. {location_context} {image_description}"
  settings:
    temperature: 0.1
""".strip(),
        encoding="utf-8",
    )

    config = load_workflow_config(config_path)

    assert config.image_analysis.model == "llava:7b"
    assert config.image_analysis.settings["image_size"] == 512
    assert config.text_analysis.model == "llama3.2:3b"


def test_load_workflow_config_requires_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
image_analysis:
  model: llava:7b
  prompt: "Describe."
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(WorkflowConfigError) as exc_info:
        load_workflow_config(config_path)

    assert "text_analysis" in str(exc_info.value)


def test_load_workflow_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text("image_analysis: [oops", encoding="utf-8")

    with pytest.raises(WorkflowConfigError):
        load_workflow_config(config_path)


def _sample_workflow_config(tmp_path: Path) -> WorkflowConfig:
    return WorkflowConfig(
        image_analysis=WorkflowStageConfig(
            model="llava:7b",
            prompt="Describe the image.{context_suffix}",
            settings={"temperature": 0.2, "image_size": 512},
        ),
        text_analysis=WorkflowStageConfig(
            model="llama3.2:3b",
            prompt="Assess it. {location_context} {image_description}",
            settings={"temperature": 0.1},
        ),
        source_path=tmp_path / "models.yaml",
    )


def _sample_result(tmp_path: Path) -> dict[str, object]:
    image_path = tmp_path / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image-bytes")
    return {
        "run_id": "abc123",
        "run_timestamp_utc": "2026-03-17T10:00:00+00:00",
        "inputs": {
            "latitude": 38.7223,
            "longitude": -9.1393,
            "zoom": 10,
            "vision_model": "llava:7b",
            "risk_model": "llama3.2:3b",
            "location_label": "Lisbon, Portugal",
        },
        "image_result": ImageDownloadResult(
            image_path=image_path,
            image_url="https://example.com/image",
            bbox=(-9.2, 38.7, -9.0, 38.8),
            generated_at_utc="2026-03-17T10:00:00+00:00",
        ),
        "description": "Dense vegetation with river channels.",
        "assessment": RiskAssessment(
            flagged=False,
            risk_level="low",
            risk_score=21,
            summary="Mostly stable conditions.",
            evidence=["Dense vegetation is visible.", "No obvious damage appears."],
            follow_up_questions=["Is the river seasonal?", "Are there recent comparison images?", "Is there protected status?"],
            raw_response='{"flagged": false}',
        ),
    }


def test_ensure_images_database_creates_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "database" / "images.csv"

    ensure_images_database(csv_path)

    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").strip().split(",") == IMAGES_CSV_FIELDS


def test_append_workflow_record_appends_rows(tmp_path: Path) -> None:
    config = _sample_workflow_config(tmp_path)
    result = _sample_result(tmp_path)
    csv_path = tmp_path / "database" / "images.csv"
    cache_key_1 = workflow_cache_key(38.7223, -9.1393, 10, config)
    cache_key_2 = workflow_cache_key(40.2033, -8.4103, 12, config)

    append_workflow_record(result, config, cache_key_1, csv_path=csv_path)
    result_2 = dict(result)
    result_2["run_id"] = "def456"
    result_2["inputs"] = dict(result["inputs"])
    result_2["inputs"]["latitude"] = 40.2033
    result_2["inputs"]["longitude"] = -8.4103
    result_2["inputs"]["zoom"] = 12
    append_workflow_record(result_2, config, cache_key_2, csv_path=csv_path)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_append_image_record_preserves_commas_quotes_and_multiline_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "database" / "images.csv"
    row = {field: "" for field in IMAGES_CSV_FIELDS}
    row.update(
        {
            "run_id": "row-1",
            "image_prompt": 'Line 1, with comma\nLine "2" kept',
            "image_description": 'Forest patch, water body, and "roads"\nSecond line',
            "text_prompt": 'Question:\n"Is there danger here?"',
            "text_summary": 'Looks "stable", but needs follow-up\nAnother line',
        }
    )

    append_image_record(row, csv_path=csv_path)

    physical_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(physical_lines) == 2

    cached = find_cached_workflow_result("missing-key", csv_path=csv_path)
    assert cached is None

    import csv as csv_module

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        stored_rows = list(csv_module.DictReader(handle))

    assert len(stored_rows) == 1
    assert stored_rows[0]["image_prompt"] == 'Line 1, with comma\\nLine "2" kept'
    assert stored_rows[0]["image_description"] == 'Forest patch, water body, and "roads"\\nSecond line'


def test_find_cached_workflow_result_returns_reconstructed_result(tmp_path: Path) -> None:
    config = _sample_workflow_config(tmp_path)
    result = _sample_result(tmp_path)
    csv_path = tmp_path / "database" / "images.csv"
    cache_key = workflow_cache_key(38.7223, -9.1393, 10, config)

    append_workflow_record(result, config, cache_key, csv_path=csv_path)

    cached = find_cached_workflow_result(cache_key, csv_path=csv_path)

    assert cached is not None
    assert cached["cached"] is True
    assert cached["description"] == "Dense vegetation with river channels."
    assert cached["assessment"].risk_level == "low"
    assert cached["image_result"].image_path.exists()


def test_find_cached_workflow_result_misses_when_key_changes(tmp_path: Path) -> None:
    config = _sample_workflow_config(tmp_path)
    result = _sample_result(tmp_path)
    csv_path = tmp_path / "database" / "images.csv"
    cache_key = workflow_cache_key(38.7223, -9.1393, 10, config)

    append_workflow_record(result, config, cache_key, csv_path=csv_path)

    changed_config = WorkflowConfig(
        image_analysis=WorkflowStageConfig(
            model="llava:13b",
            prompt=config.image_analysis.prompt,
            settings=config.image_analysis.settings,
        ),
        text_analysis=config.text_analysis,
        source_path=config.source_path,
    )
    changed_key = workflow_cache_key(38.7223, -9.1393, 10, changed_config)

    assert changed_key != cache_key
    assert find_cached_workflow_result(changed_key, csv_path=csv_path) is None


def test_execute_governed_ai_workflow_uses_cache_before_ai_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
image_analysis:
  model: llava:7b
  prompt: "Describe the image.{context_suffix}"
  settings:
    temperature: 0.2
    image_size: 512
text_analysis:
  model: llama3.2:3b
  prompt: "Assess it. {location_context} {image_description}"
  settings:
    temperature: 0.1
""".strip(),
        encoding="utf-8",
    )
    config = load_workflow_config(config_path)
    result = _sample_result(tmp_path)
    csv_path = tmp_path / "database" / "images.csv"
    cache_key = workflow_cache_key(38.7223, -9.1393, 10, config)
    append_workflow_record(result, config, cache_key, csv_path=csv_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AI pipeline should not run when cache is available")

    monkeypatch.setattr("app.ai_workflow.check_ollama_available", fail_if_called)
    monkeypatch.setattr("app.ai_workflow.download_esri_world_imagery", fail_if_called)

    cached = execute_governed_ai_workflow(
        38.7223,
        -9.1393,
        10,
        location_label="Lisbon, Portugal",
        images_dir=tmp_path / "images",
        config_path=config_path,
        database_csv_path=csv_path,
    )

    assert cached["cached"] is True
    assert cached["cache_key"] == cache_key


def test_settings_fingerprint_changes_with_config(tmp_path: Path) -> None:
    config = _sample_workflow_config(tmp_path)
    changed_config = WorkflowConfig(
        image_analysis=WorkflowStageConfig(
            model=config.image_analysis.model,
            prompt="Different prompt.{context_suffix}",
            settings=config.image_analysis.settings,
        ),
        text_analysis=config.text_analysis,
        source_path=config.source_path,
    )

    assert settings_fingerprint(config) != settings_fingerprint(changed_config)
