from __future__ import annotations

from app.ai_workflow import build_bbox, extract_json_object


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
