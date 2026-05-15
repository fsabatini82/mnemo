"""Tests for the LLM-powered deep drift detection (F5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnemo.audit_deep import DeepAuditEngine
from mnemo.core.models import Hit, QueryResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def ingest(self, documents) -> None:  # pragma: no cover
        pass

    def query(self, question: str, *, k: int = 5, where=None) -> QueryResult:
        return QueryResult(
            question=question,
            hits=[h for h in self._hits if h.doc_id == question][:k],
        )


class _FakeRunner:
    """Mocks CopilotRunner — records prompts and returns canned JSON."""

    def __init__(self, response: Any, available: bool = True) -> None:
        self._response = response
        self._available = available
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def describe(self) -> str:
        return "FakeRunner()"

    def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self._response, str):
            return self._response
        return json.dumps(self._response)

    def run_json(self, prompt: str) -> Any:
        from mnemo.ingestion.agents.copilot.runner import extract_json
        return extract_json(self.run(prompt))


def _impl_hit(spec_id: str, *, related_files: str, **extra: Any) -> Hit:
    return Hit(
        chunk_id=f"{spec_id}::chunk-0000",
        doc_id=spec_id,
        text="...",
        score=1.0,
        metadata={
            "kind": "story",
            "lifecycle": "implemented",
            "related_files": related_files,
            "template_compliance": "full",
            "acceptance_summary": "Endpoint X filters by Y",
            "acceptance_criteria": "- filter Y is honored",
            "test_scenarios_happy": "Given a request When Y=Z Then filtered",
            "test_scenarios_error": "",
            "test_scenarios_edge": "",
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# Engine behavior
# ---------------------------------------------------------------------------


def test_skips_non_implemented_specs(tmp_path: Path) -> None:
    hit = _impl_hit("US-1", related_files="x.py")
    hit.metadata["lifecycle"] = "proposed"
    runner = _FakeRunner({"aligned": False, "severity": "high"})
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    assert engine.audit_spec("US-1") == []
    assert runner.prompts == []  # never called the LLM


def test_skips_when_no_files_can_be_read(tmp_path: Path) -> None:
    hit = _impl_hit("US-1", related_files="missing/from/disk.py")
    runner = _FakeRunner({"aligned": False, "severity": "high"})
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    assert engine.audit_spec("US-1") == []
    assert runner.prompts == []


def test_aligned_returns_no_issues(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("# US-1\n", encoding="utf-8")
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner({
        "aligned": True, "severity": "none", "confidence": "high",
        "divergences": [], "summary": "Code matches spec",
    })
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    issues = engine.audit_spec("US-1")
    assert issues == []


def test_high_drift_issue_produced(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("def f(): pass\n", encoding="utf-8")
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner({
        "aligned": False,
        "severity": "high",
        "confidence": "high",
        "divergences": [
            {"file": "x.py", "description": "filter not applied",
             "evidence": "def f(): pass"},
        ],
        "summary": "Spec says filter Y, code ignores it.",
    })
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    issues = engine.audit_spec("US-1")
    assert len(issues) == 1
    issue = issues[0]
    assert issue.type == "behavior"
    assert issue.severity == "high"
    assert "filter" in issue.description.lower()
    assert issue.details["divergences"][0]["file"] == "x.py"
    assert issue.details["confidence"] == "high"


def test_invalid_severity_coerced_to_none(tmp_path: Path) -> None:
    """If the LLM returns garbage severity, coerce safely to 'none'."""
    (tmp_path / "x.py").write_text("body\n", encoding="utf-8")
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner({
        "aligned": False,
        "severity": "catastrophic",  # not in enum
        "confidence": "high",
        "divergences": [],
        "summary": "x",
    })
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    issues = engine.audit_spec("US-1")
    # Severity coerced to "none" + aligned=False → still no issue because severity=none
    assert issues == []


def test_garbage_json_yields_no_issues(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("body\n", encoding="utf-8")
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner("the model didn't return JSON at all")
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    assert engine.audit_spec("US-1") == []


def test_runner_unavailable_short_circuits(tmp_path: Path) -> None:
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner({"aligned": True}, available=False)
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    assert engine.is_available() is False


def test_truncates_large_files(tmp_path: Path) -> None:
    big = "X" * 20000
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    hit = _impl_hit("US-1", related_files="big.py")
    runner = _FakeRunner({"aligned": True, "severity": "none",
                          "confidence": "high", "divergences": []})
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path,
                             runner=runner, file_budget=1000)
    engine.audit_spec("US-1")
    assert "truncated" in runner.prompts[0]


def test_unknown_spec_returns_empty(tmp_path: Path) -> None:
    runner = _FakeRunner({"aligned": True})
    engine = DeepAuditEngine(_FakePipeline([]), code_root=tmp_path, runner=runner)
    assert engine.audit_spec("US-DOESNT-EXIST") == []
    assert runner.prompts == []


def test_evidence_truncated_to_1000_chars(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("body\n", encoding="utf-8")
    hit = _impl_hit("US-1", related_files="x.py")
    runner = _FakeRunner({
        "aligned": False, "severity": "medium", "confidence": "high",
        "divergences": [
            {"file": "x.py", "description": "d", "evidence": "Z" * 5000},
        ],
        "summary": "x",
    })
    engine = DeepAuditEngine(_FakePipeline([hit]), code_root=tmp_path, runner=runner)
    issues = engine.audit_spec("US-1")
    assert len(issues[0].details["divergences"][0]["evidence"]) <= 1000
