"""Tests for the factory that wires Settings → MnemoSystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemo.config import Settings
from mnemo.factory import MnemoSystem, _build_store, build_pipeline, build_system


class _FakeEmbedder:
    dimension = 8

    def __init__(self, model_name: str = "x") -> None:
        self.model_name = model_name

    def embed(self, texts: Any) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy FastEmbedEmbedder with a deterministic stub."""
    monkeypatch.setattr(
        "mnemo.embedders.fastembed_embedder.FastEmbedEmbedder",
        _FakeEmbedder,
    )


def test_build_system_with_chroma_default(
    patched_embedder: None, tmp_path: Path
) -> None:
    settings = Settings(
        store="chroma",
        pipeline="default",
        persist_dir=tmp_path / "data",
    )
    system = build_system(settings)
    assert isinstance(system, MnemoSystem)
    assert system.settings is settings
    # Both pipelines exist and are independent objects.
    assert system.specs is not None
    assert system.bugs is not None
    assert system.specs is not system.bugs


def test_build_pipeline_alias_returns_specs_pipeline(
    patched_embedder: None, tmp_path: Path
) -> None:
    settings = Settings(persist_dir=tmp_path / "data")
    pipeline = build_pipeline(settings)
    # The convenience alias returns the specs pipeline.
    assert pipeline is not None


def test_build_system_lance_without_extras_raises(
    patched_embedder: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        store="lance",
        pipeline="default",
        persist_dir=tmp_path / "data",
    )
    # Simulate `mnemo[lance]` extras NOT being installed.
    import builtins

    real_import = builtins.__import__

    def faux_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mnemo.stores.lance_store":
            raise ImportError("simulated missing lancedb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    with pytest.raises(RuntimeError, match="lance"):
        build_system(settings)


def test_build_system_llamaindex_without_extras_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        pipeline="llamaindex",
        persist_dir=tmp_path / "data",
    )
    import builtins

    real_import = builtins.__import__

    def faux_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mnemo.pipelines.llamaindex":
            raise ImportError("simulated missing llama-index")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    with pytest.raises(RuntimeError, match="LlamaIndex"):
        build_system(settings)


def test_build_store_unknown_raises(tmp_path: Path) -> None:
    settings = Settings(persist_dir=tmp_path / "data")
    # Bypass Settings validation by mutating the underlying dict directly.
    object.__setattr__(settings, "store", "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown store"):
        _build_store(settings, dimension=4, collection="x")
