"""Tests for the factory that wires Settings → MnemoSystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemo.config import Settings
from mnemo.factory import (
    MnemoSystem,
    _build_store,
    _ephemeral_id,
    build_pipeline,
    build_system,
)
from mnemo.registry import open_registry


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


# ---------------------------------------------------------------------------
# Default build
# ---------------------------------------------------------------------------


def test_build_system_with_chroma_default(
    patched_embedder: None, tmp_path: Path,
) -> None:
    settings = Settings(
        store="chroma",
        pipeline="default",
        persist_dir=tmp_path / "data",
        project="alpha",
        environment="dev",
    )
    system = build_system(settings, auto_register=True)
    assert isinstance(system, MnemoSystem)
    assert system.settings is settings
    assert system.specs is not None
    assert system.bugs is not None
    assert system.specs is not system.bugs
    # New: project_id and effective prefix are surfaced on the system.
    assert system.project_id == "001"
    assert system.environment == "dev"
    assert system.effective_prefix == "001_dev"


def test_build_pipeline_alias_returns_specs_pipeline(
    patched_embedder: None, tmp_path: Path,
) -> None:
    settings = Settings(persist_dir=tmp_path / "data")
    pipeline = build_pipeline(settings)
    assert pipeline is not None


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------


def test_build_system_uses_ephemeral_id_when_project_unknown(
    patched_embedder: None, tmp_path: Path,
) -> None:
    settings = Settings(
        persist_dir=tmp_path / "data",
        project="unknown-yet",
        environment="dev",
    )
    # auto_register=False — read-only path used by the MCP server.
    system = build_system(settings, auto_register=False)
    # Ephemeral id is deterministic for a given slug.
    assert system.project_id == _ephemeral_id("unknown-yet")
    # Registry not touched.
    registry = open_registry(tmp_path / "data")
    assert len(registry) == 0


def test_build_system_auto_registers_when_requested(
    patched_embedder: None, tmp_path: Path,
) -> None:
    settings = Settings(
        persist_dir=tmp_path / "data",
        project="new-proj",
        environment="prd",
    )
    system = build_system(settings, auto_register=True)
    assert system.project_id == "001"

    registry = open_registry(tmp_path / "data")
    record = registry.get("new-proj")
    assert record is not None
    assert record.id == "001"
    assert "prd" in record.environments


def test_build_system_preserves_id_across_envs(
    patched_embedder: None, tmp_path: Path,
) -> None:
    """Same project across different envs keeps the same id, different prefix."""
    s_dev = Settings(persist_dir=tmp_path / "data", project="alpha", environment="dev")
    s_prd = Settings(persist_dir=tmp_path / "data", project="alpha", environment="prd")
    sys_dev = build_system(s_dev, auto_register=True)
    sys_prd = build_system(s_prd, auto_register=True)
    assert sys_dev.project_id == sys_prd.project_id == "001"
    assert sys_dev.effective_prefix == "001_dev"
    assert sys_prd.effective_prefix == "001_prd"


def test_ephemeral_id_is_deterministic_and_three_digits() -> None:
    ids = {_ephemeral_id(s) for s in ["alpha", "beta", "demo-project", "a", "z" * 32]}
    assert all(len(i) == 3 and i.isdigit() for i in ids)


# ---------------------------------------------------------------------------
# Backend missing extras
# ---------------------------------------------------------------------------


def test_build_system_lance_without_extras_raises(
    patched_embedder: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = Settings(
        store="lance",
        pipeline="default",
        persist_dir=tmp_path / "data",
    )
    import builtins

    real_import = builtins.__import__

    def faux_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mnemo.stores.lance_store":
            raise ImportError("simulated missing lancedb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    with pytest.raises(RuntimeError, match="lance"):
        build_system(settings, auto_register=True)


def test_build_system_llamaindex_without_extras_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
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
        build_system(settings, auto_register=True)


def test_build_store_unknown_raises(tmp_path: Path) -> None:
    settings = Settings(persist_dir=tmp_path / "data")
    object.__setattr__(settings, "store", "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown store"):
        _build_store(settings, dimension=4, collection="x")
