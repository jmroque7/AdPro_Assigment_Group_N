from __future__ import annotations

from pathlib import Path
import pytest

from app.okavango import download_all_datasets


class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_download_all_datasets_creates_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    def fake_get(url: str, timeout: int = 60) -> DummyResponse:
        # return different bytes for different URLs
        return DummyResponse(f"from,{url}\n1,2\n".encode("utf-8"), 200)

    monkeypatch.setattr(requests, "get", fake_get)

    urls = {"a.csv": "https://example.com/a", "b.csv": "https://example.com/b"}

    download_all_datasets(tmp_path, urls)

    assert (tmp_path / "a.csv").exists()
    assert (tmp_path / "b.csv").exists()
    assert (tmp_path / "a.csv").stat().st_size > 0
    assert (tmp_path / "b.csv").stat().st_size > 0


def test_download_all_datasets_skips_existing_nonempty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    calls: list[str] = []

    def fake_get(url: str, timeout: int = 60) -> DummyResponse:
        calls.append(url)
        return DummyResponse(b"x\n", 200)

    monkeypatch.setattr(requests, "get", fake_get)

    # precreate a.csv so it should be skipped
    (tmp_path / "a.csv").write_bytes(b"already-here")

    urls = {"a.csv": "https://example.com/a", "b.csv": "https://example.com/b"}

    download_all_datasets(tmp_path, urls)

    assert "https://example.com/a" not in calls
    assert "https://example.com/b" in calls