"""Tests for the Settings configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemo.config import Settings, load_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Isolate env so user's real env vars don't leak.
    for k in list(monkeypatch._setitem):  # type: ignore[attr-defined]
        pass
    monkeypatch.chdir(tmp_path)
    # Clear any MNEMO_* env vars in scope
    monkeypatch.delenv("MNEMO_STORE", raising=False)
    settings = Settings()
    assert settings.store == "chroma"
    assert settings.pipeline == "default"
    assert settings.specs_collection == "specs"
    assert settings.bugs_collection == "bug_memory"
    assert settings.top_k == 5


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMO_STORE", "lance")
    monkeypatch.setenv("MNEMO_TOP_K", "10")
    monkeypatch.setenv("MNEMO_CHUNK_SIZE", "256")
    settings = load_settings()
    assert settings.store == "lance"
    assert settings.top_k == 10
    assert settings.chunk_size == 256


def test_invalid_chunk_size_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMO_CHUNK_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_negative_chunk_overlap_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMO_CHUNK_OVERLAP", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_store_value_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMO_STORE", "redis")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_pipeline_value_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MNEMO_PIPELINE", "magic")
    with pytest.raises(ValidationError):
        Settings()


def test_load_settings_returns_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert isinstance(settings, Settings)
