"""Shared pytest fixtures for the Mnemo test suite."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from mnemo.config import Settings
from mnemo.core.models import Chunk, Document


# ---------------------------------------------------------------------------
# Test isolation — neutralize the user's MNEMO_* env vars so tests don't
# inherit ambient config from the shell.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_mnemo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("MNEMO_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Fake embedder — deterministic, no model download.
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Embedder stub that produces deterministic, hash-based vectors."""

    dimension: int = 8

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Deterministic per-text vector based on character codes — enough
        # to give different texts different vectors, without needing a
        # real embedding model.
        result: list[list[float]] = []
        for t in texts:
            base = [ord(c) for c in (t or "_")[: self.dimension]]
            padded = (base + [0] * self.dimension)[: self.dimension]
            norm = sum(v * v for v in padded) ** 0.5 or 1.0
            result.append([v / norm for v in padded])
        return result


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Tiny corpora on disk — used by ingestion + integration-ish tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def specs_dir(tmp_path: Path) -> Path:
    """Build a small specs/ tree mimicking the lab fixture layout."""
    root = tmp_path / "specs"
    (root / "epics").mkdir(parents=True)
    (root / "stories").mkdir(parents=True)
    (root / "adrs").mkdir(parents=True)

    (root / "epics" / "EPIC-BE.md").write_text(
        "---\nid: EPIC-BE\ntitle: Backend\n---\n\nBackend epic body.\n",
        encoding="utf-8",
    )
    (root / "stories" / "US-001.md").write_text(
        "---\nid: US-001\ntitle: Test story\nepic: EPIC-BE\n"
        "related_bugs: [BUG-1]\nrelated_adrs: [ADR-1]\n---\n\n"
        "## Acceptance\nDo the thing.\n",
        encoding="utf-8",
    )
    (root / "adrs" / "ADR-001.md").write_text(
        "---\nid: ADR-001\ntitle: Some decision\n---\n\nDecide stuff.\n",
        encoding="utf-8",
    )
    # File without frontmatter — exercises the fallback branch.
    (root / "stories" / "raw.md").write_text(
        "# Raw spec\nNo frontmatter here.\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def bugs_dir(tmp_path: Path) -> Path:
    """Build a small bugs/ tree with one fully-specified record + one minimal."""
    root = tmp_path / "bugs"
    root.mkdir(parents=True)

    full = {
        "id": "BUG-001",
        "title": "Sample bug",
        "severity": "high",
        "status": "resolved",
        "epic": "EPIC-BE",
        "symptom": "Something fails",
        "root_cause": "A typo in X",
        "fix_summary": "Fix the typo",
        "files_touched": ["a.py", "b.py"],
        "pattern_tags": ["typo", "regression"],
        "related_spec": "US-001",
        "related_pr": "https://example.com/pr/1",
        "resolved": "2025-01-01",
        "resolved_by": "x@y.com",
    }
    (root / "BUG-001.json").write_text(json.dumps(full), encoding="utf-8")
    (root / "BUG-002.json").write_text(
        json.dumps({"id": "BUG-002", "title": "Minimal record"}),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Sample in-memory data — used by chunking / pipeline tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(id="doc-1", text="Hello world. " * 30, metadata={"source": "a"}),
        Document(id="doc-2", text="Second document body. " * 30, metadata={"source": "b"}),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(id="c-1", doc_id="doc-1", text="alpha", metadata={"i": 0}),
        Chunk(id="c-2", doc_id="doc-1", text="beta", metadata={"i": 1}),
    ]


# ---------------------------------------------------------------------------
# Settings fixture pointing at an isolated tmp persist dir.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_settings(tmp_path: Path, specs_dir: Path, bugs_dir: Path) -> Settings:
    """Settings with all paths pointed at tmp directories."""
    return Settings(
        persist_dir=tmp_path / "data",
        specs_source_dir=specs_dir,
        bugs_source_dir=bugs_dir,
        chunk_size=128,
        chunk_overlap=16,
        top_k=3,
    )
