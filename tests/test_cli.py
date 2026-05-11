"""Tests for the mnemo-ingest CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import mnemo.cli as cli
from mnemo.core.models import Chunk, Document, Hit, QueryResult
from mnemo.factory import MnemoSystem


class _RecordingPipeline:
    """RagPipeline stub that records what was ingested."""

    def __init__(self) -> None:
        self.ingested: list[list[Document]] = []

    def ingest(self, documents: Sequence[Document]) -> None:
        self.ingested.append(list(documents))

    def query(self, question: str, *, k: int = 5) -> QueryResult:
        return QueryResult(question=question, hits=[])


@pytest.fixture
def patched_system(
    monkeypatch: pytest.MonkeyPatch,
    specs_dir: Path,
    bugs_dir: Path,
    tmp_path: Path,
) -> tuple[_RecordingPipeline, _RecordingPipeline]:
    """Replace build_system with one that returns recording pipelines."""
    specs_pipeline = _RecordingPipeline()
    bugs_pipeline = _RecordingPipeline()

    def fake_build(settings: Any) -> MnemoSystem:
        return MnemoSystem(specs=specs_pipeline, bugs=bugs_pipeline, settings=settings)

    monkeypatch.setattr("mnemo.cli.build_system", fake_build)
    # Default-arg paths read from Settings; point them at our fixtures.
    monkeypatch.setenv("MNEMO_SPECS_SOURCE_DIR", str(specs_dir))
    monkeypatch.setenv("MNEMO_BUGS_SOURCE_DIR", str(bugs_dir))
    monkeypatch.setenv("MNEMO_PERSIST_DIR", str(tmp_path / "data"))
    return specs_pipeline, bugs_pipeline


def test_specs_subcommand_ingests_documents(patched_system) -> None:
    specs_pipeline, _ = patched_system
    rc = cli.main(["specs"])
    assert rc == 0
    assert len(specs_pipeline.ingested) == 1
    assert len(specs_pipeline.ingested[0]) == 4


def test_bugs_subcommand_ingests_documents(patched_system) -> None:
    _, bugs_pipeline = patched_system
    rc = cli.main(["bugs"])
    assert rc == 0
    assert len(bugs_pipeline.ingested) == 1
    assert len(bugs_pipeline.ingested[0]) == 2


def test_all_subcommand_ingests_both(patched_system) -> None:
    specs_pipeline, bugs_pipeline = patched_system
    rc = cli.main(["all"])
    assert rc == 0
    assert len(specs_pipeline.ingested) == 1
    assert len(bugs_pipeline.ingested) == 1


def test_specs_with_explicit_path(
    patched_system,
    specs_dir: Path,
) -> None:
    specs_pipeline, _ = patched_system
    rc = cli.main(["specs", "--path", str(specs_dir)])
    assert rc == 0
    assert specs_pipeline.ingested[0]


def test_specs_with_invalid_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patched_system,
) -> None:
    missing = tmp_path / "no-such"
    with pytest.raises(FileNotFoundError):
        cli.main(["specs", "--path", str(missing)])


def test_ingest_helper_warns_on_empty_list() -> None:
    """The internal helper returns nonzero when there's nothing to ingest."""
    pipeline = _RecordingPipeline()
    rc = cli._ingest(pipeline, [], "specs")  # type: ignore[arg-type]
    assert rc == 1
    assert pipeline.ingested == []


def test_ingest_helper_returns_zero_when_documents_present() -> None:
    pipeline = _RecordingPipeline()
    docs = [Document(id="d", text="t")]
    rc = cli._ingest(pipeline, docs, "specs")
    assert rc == 0
    assert pipeline.ingested == [docs]


def test_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
