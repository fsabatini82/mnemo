"""Tests for the Copilot-CLI-backed ingestion agents.

The real Copilot CLI is *not* invoked here — we mock the `CopilotRunner`
so the suite stays deterministic, fast, and offline. The integration
test against a live CLI would be a separate manual smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnemo.ingestion.agents.copilot.bugs_agent import CopilotBugsAgent
from mnemo.ingestion.agents.copilot.runner import (
    CopilotRunner,
    CopilotRunnerError,
    extract_json,
)
from mnemo.ingestion.agents.copilot.specs_agent import CopilotSpecsAgent


# ---------------------------------------------------------------------------
# extract_json helper
# ---------------------------------------------------------------------------


def test_extract_json_whole_string() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block() -> None:
    text = "Here is the result:\n```json\n{\"a\": 2}\n```\nThanks."
    assert extract_json(text) == {"a": 2}


def test_extract_json_from_inline_braces() -> None:
    text = 'preamble {"a": 3, "b": [1,2]} trailing words'
    assert extract_json(text) == {"a": 3, "b": [1, 2]}


def test_extract_json_returns_none_on_garbage() -> None:
    assert extract_json("no json here at all") is None


def test_extract_json_returns_none_on_empty() -> None:
    assert extract_json("") is None


def test_extract_json_returns_none_on_null_literal() -> None:
    assert extract_json("null") is None


# ---------------------------------------------------------------------------
# Fake runner — drives the agent without touching subprocess
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Drop-in stand-in for CopilotRunner.

    Holds a mapping of "prompt fingerprint → JSON response" so each
    test can declare exactly what the LLM would have answered for each
    input file.
    """

    def __init__(self, responses_by_marker: dict[str, Any]) -> None:
        self._responses = responses_by_marker
        self.prompts_seen: list[str] = []

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return "FakeRunner()"

    def run(self, prompt: str) -> str:
        # Pick the first marker that appears in the prompt.
        for marker, response in self._responses.items():
            if marker in prompt:
                self.prompts_seen.append(marker)
                if isinstance(response, str):
                    return response
                return json.dumps(response)
        raise AssertionError(f"No fake response configured for prompt:\n{prompt[:200]}")

    def run_json(self, prompt: str) -> Any:
        raw = self.run(prompt)
        return extract_json(raw)


# ---------------------------------------------------------------------------
# Specs agent
# ---------------------------------------------------------------------------


@pytest.fixture
def specs_tmp_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "specs"
    (root / "stories").mkdir(parents=True)
    (root / "stories" / "US-001.md").write_text(
        "---\nid: US-001\ntitle: Indexable story\n---\n\nAcceptance: do X.\n",
        encoding="utf-8",
    )
    (root / "stories" / "scratch.md").write_text(
        "# scratch notes\nrandom thoughts that aren't a spec.\n",
        encoding="utf-8",
    )
    (root / "adrs").mkdir(parents=True)
    (root / "adrs" / "ADR-001.md").write_text(
        "---\nid: ADR-001\ntitle: A decision\n---\n\nWe decided X.\n",
        encoding="utf-8",
    )
    return root


def test_specs_agent_indexes_valid_specs(specs_tmp_corpus: Path) -> None:
    runner = _FakeRunner(
        responses_by_marker={
            "US-001.md": {
                "indexable": True,
                "id": "US-001",
                "title": "Indexable story",
                "kind": "story",
                "epic": "EPIC-X",
                "status": "ready",
                "related_bugs": ["BUG-1"],
                "related_adrs": [],
                "related_files": [],
                "body": "Acceptance: do X.",
            },
            "ADR-001.md": {
                "indexable": True,
                "id": "ADR-001",
                "title": "A decision",
                "kind": "adr",
                "epic": None,
                "status": "accepted",
                "related_bugs": [],
                "related_adrs": [],
                "related_files": [],
                "body": "We decided X.",
            },
            "scratch.md": {"indexable": False, "id": None},
        }
    )
    agent = CopilotSpecsAgent(runner=runner)  # type: ignore[arg-type]
    docs = agent.ingest(specs_tmp_corpus)

    # Two indexable items, scratch.md skipped as noise.
    assert {d.id for d in docs} == {"US-001", "ADR-001"}
    by_id = {d.id: d for d in docs}
    assert by_id["US-001"].metadata["title"] == "Indexable story"
    assert by_id["US-001"].metadata["epic"] == "EPIC-X"
    assert by_id["US-001"].metadata["related_bugs"] == "BUG-1"
    assert by_id["US-001"].metadata["extracted_by"] == "copilot-cli"
    assert "Indexable story" in by_id["US-001"].text


def test_specs_agent_skips_when_no_json(specs_tmp_corpus: Path) -> None:
    runner = _FakeRunner(
        responses_by_marker={
            "US-001.md": "garbage non-json output from the model",
            "ADR-001.md": "more garbage",
            "scratch.md": "still garbage",
        }
    )
    agent = CopilotSpecsAgent(runner=runner)  # type: ignore[arg-type]
    docs = agent.ingest(specs_tmp_corpus)
    assert docs == []


def test_specs_agent_continues_after_runner_error(
    specs_tmp_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    class _FlakyRunner(_FakeRunner):
        def run_json(self, prompt: str) -> Any:
            calls["n"] += 1
            if "ADR-001.md" in prompt:
                raise CopilotRunnerError("simulated transient failure")
            return super().run_json(prompt)

    runner = _FlakyRunner(
        responses_by_marker={
            "US-001.md": {"indexable": True, "id": "US-001", "title": "X", "kind": "story", "body": "Body"},
            "scratch.md": {"indexable": False, "id": None},
        }
    )
    agent = CopilotSpecsAgent(runner=runner)  # type: ignore[arg-type]
    docs = agent.ingest(specs_tmp_corpus)

    # ADR-001 fails, US-001 succeeds, scratch is noise → only US-001 indexed.
    assert {d.id for d in docs} == {"US-001"}
    assert calls["n"] >= 2


def test_specs_agent_raises_on_missing_dir(tmp_path: Path) -> None:
    runner = _FakeRunner(responses_by_marker={})
    agent = CopilotSpecsAgent(runner=runner)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        agent.ingest(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# Bugs agent
# ---------------------------------------------------------------------------


@pytest.fixture
def bugs_tmp_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "bugs"
    root.mkdir()
    (root / "BUG-1.json").write_text(
        json.dumps({"id": "BUG-1", "title": "Real bug", "symptom": "X breaks"}),
        encoding="utf-8",
    )
    (root / "BUG-2.json").write_text(
        json.dumps({"id": "BUG-2", "title": "Typo fix"}),
        encoding="utf-8",
    )
    (root / "broken.json").write_text("this is not json", encoding="utf-8")
    return root


def test_bugs_agent_indexes_real_bugs(bugs_tmp_corpus: Path) -> None:
    runner = _FakeRunner(
        responses_by_marker={
            "BUG-1.json": {
                "indexable": True,
                "id": "BUG-1",
                "title": "Real bug",
                "severity": "high",
                "status": "resolved",
                "epic": "EPIC-BE",
                "symptom": "X breaks under load.",
                "root_cause": "Missing index on T(a,b).",
                "fix_summary": "Add composite index.",
                "files_touched": ["migrations/001.sql"],
                "pattern_tags": ["missing-index", "perf"],
                "related_spec": "US-100",
                "related_pr": "https://example.com/pr/1",
                "resolved": "2025-10-01",
            },
            "BUG-2.json": {"indexable": False, "id": "BUG-2"},
        }
    )
    agent = CopilotBugsAgent(runner=runner)  # type: ignore[arg-type]
    docs = agent.ingest(bugs_tmp_corpus)

    # BUG-1 indexed, BUG-2 noise, broken.json never reaches the LLM.
    assert {d.id for d in docs} == {"BUG-1"}
    bug = docs[0]
    assert bug.metadata["severity"] == "high"
    assert bug.metadata["pattern_tags"] == "missing-index, perf"
    assert bug.metadata["files_touched"] == "migrations/001.sql"
    assert bug.metadata["extracted_by"] == "copilot-cli"
    assert "Missing index" in bug.text


def test_bugs_agent_skips_invalid_json_files(bugs_tmp_corpus: Path) -> None:
    runner = _FakeRunner(
        responses_by_marker={
            "BUG-1.json": {"indexable": False, "id": "BUG-1"},
            "BUG-2.json": {"indexable": False, "id": "BUG-2"},
            # broken.json must NOT be passed to the LLM at all.
        }
    )
    agent = CopilotBugsAgent(runner=runner)  # type: ignore[arg-type]
    docs = agent.ingest(bugs_tmp_corpus)
    assert docs == []
    # The runner should have seen at most BUG-1 and BUG-2 markers, never "broken".
    assert all(m in {"BUG-1.json", "BUG-2.json"} for m in runner.prompts_seen)


# ---------------------------------------------------------------------------
# Runner — light-touch tests on plumbing (no real subprocess)
# ---------------------------------------------------------------------------


def test_runner_describe_includes_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMO_COPILOT_BIN", "/fake/copilot")
    monkeypatch.setenv("MNEMO_COPILOT_ARGS", "--no-stream --foo")
    runner = CopilotRunner()
    desc = runner.describe()
    assert "/fake/copilot" in desc
    assert "--no-stream" in desc and "--foo" in desc


def test_runner_run_raises_when_binary_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMO_COPILOT_BIN", "/definitely/not/a/real/binary/copilot")
    runner = CopilotRunner()
    with pytest.raises(CopilotRunnerError, match="not found"):
        runner.run("hi")
