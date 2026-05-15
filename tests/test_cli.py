"""Tests for the mnemo-ingest CLI."""

from __future__ import annotations

import os
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

    def fake_build(settings: Any, **kwargs: Any) -> MnemoSystem:
        return MnemoSystem(
            specs=specs_pipeline,
            bugs=bugs_pipeline,
            settings=settings,
            project_id="999",
            environment=settings.environment,
        )

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


# ---------------------------------------------------------------------------
# Project/env flags
# ---------------------------------------------------------------------------


def test_specs_flag_project_propagates_to_env(patched_system) -> None:
    """--project / --env are applied to os.environ before Settings load."""
    rc = cli.main(["specs", "--project", "alpha", "--env", "prd"])
    assert rc == 0
    assert os.environ.get("MNEMO_PROJECT") == "alpha"
    assert os.environ.get("MNEMO_ENVIRONMENT") == "prd"


def test_specs_rejects_invalid_project_slug() -> None:
    with pytest.raises(SystemExit):
        cli.main(["specs", "--project", "Alpha"])  # uppercase = invalid


def test_specs_rejects_invalid_environment() -> None:
    with pytest.raises(SystemExit):
        cli.main(["specs", "--env", "uat"])  # not in enum


# ---------------------------------------------------------------------------
# --agentic semantics (F6) + --reasoning-effort
# ---------------------------------------------------------------------------


def test_specs_agentic_bare_defaults_to_gpt5_mini(
    patched_system, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--agentic` with no value should build a gpt-5-mini runner."""
    seen: dict[str, str] = {}

    class _FakeRunner:
        def is_available(self): return True
        def describe(self): return "Fake(openai/gpt-5-mini)"

    def fake_build(value, settings, *, reasoning_effort=None):
        seen["agentic"] = value
        seen["reasoning"] = reasoning_effort
        return _FakeRunner()

    monkeypatch.setattr("mnemo.cli.build_runner", fake_build)

    # Use a do-nothing agent that doesn't actually call the runner.
    class _NoopAgent:
        def __init__(self, runner): pass
        def ingest(self, source): return []

    monkeypatch.setattr("mnemo.ingestion.agents.copilot.specs_agent.CopilotSpecsAgent",
                       _NoopAgent)
    # No docs means rc=1 (warning), but we just care about the flag plumbing.
    cli.main(["specs", "--agentic"])
    assert seen["agentic"] == "gpt-5-mini"
    # Task-default for ingest is "low"
    assert seen["reasoning"] == "low"


def test_specs_agentic_with_explicit_model(
    patched_system, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    class _FakeRunner:
        def is_available(self): return True
        def describe(self): return "Fake"

    def fake_build(value, settings, *, reasoning_effort=None):
        seen["agentic"] = value
        return _FakeRunner()

    monkeypatch.setattr("mnemo.cli.build_runner", fake_build)
    monkeypatch.setattr("mnemo.ingestion.agents.copilot.specs_agent.CopilotSpecsAgent",
                       type("A", (), {
                           "__init__": lambda self, runner: None,
                           "ingest": lambda self, source: [],
                       }))
    cli.main(["specs", "--agentic", "gpt-5"])
    assert seen["agentic"] == "gpt-5"


def test_specs_reasoning_effort_override(
    patched_system, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--reasoning-effort overrides the task default."""
    seen: dict[str, str] = {}

    class _FakeRunner:
        def is_available(self): return True
        def describe(self): return "Fake"

    def fake_build(value, settings, *, reasoning_effort=None):
        seen["reasoning"] = reasoning_effort
        return _FakeRunner()

    monkeypatch.setattr("mnemo.cli.build_runner", fake_build)
    monkeypatch.setattr("mnemo.ingestion.agents.copilot.specs_agent.CopilotSpecsAgent",
                       type("A", (), {
                           "__init__": lambda self, runner: None,
                           "ingest": lambda self, source: [],
                       }))
    cli.main(["specs", "--agentic", "gpt-5", "--reasoning-effort", "high"])
    assert seen["reasoning"] == "high"


def test_specs_agentic_unavailable_runner_exits(
    patched_system, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DeadRunner:
        def is_available(self): return False
        def describe(self): return "DeadRunner()"

    monkeypatch.setattr(
        "mnemo.cli.build_runner",
        lambda *a, **kw: _DeadRunner(),
    )
    with pytest.raises(SystemExit):
        cli.main(["specs", "--agentic", "gpt-5"])
