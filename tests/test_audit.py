"""Tests for the drift-detection audit engine (F4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnemo.audit import (
    AuditEngine,
    DriftIssue,
    DriftReport,
    _parse_csv,
    _scan_codebase_for_id,
    _spec_id_from_chunk,
    _worst_severity,
)
from mnemo.core.models import Hit, QueryResult


# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------


def test_parse_csv_empty() -> None:
    assert _parse_csv(None) == []
    assert _parse_csv("") == []


def test_parse_csv_from_string() -> None:
    assert _parse_csv("a.py, b.py , c.py") == ["a.py", "b.py", "c.py"]


def test_parse_csv_from_list() -> None:
    assert _parse_csv(["a", " b ", ""]) == ["a", "b"]


def test_spec_id_from_chunk() -> None:
    assert _spec_id_from_chunk("US-102::chunk-0000") == "US-102"
    assert _spec_id_from_chunk("plain-id") == "plain-id"


def test_worst_severity_picks_highest() -> None:
    issues = [
        DriftIssue("a", "low", "x", "y"),
        DriftIssue("b", "high", "x", "y"),
        DriftIssue("c", "medium", "x", "y"),
    ]
    assert _worst_severity(issues) == "high"


def test_worst_severity_none_for_empty() -> None:
    assert _worst_severity([]) == "none"


# ---------------------------------------------------------------------------
# Codebase scanner
# ---------------------------------------------------------------------------


def test_scan_codebase_finds_id(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("# US-102 implementation\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("nothing relevant\n", encoding="utf-8")
    (tmp_path / "src" / "c.md").write_text("see US-102 for context", encoding="utf-8")
    found = _scan_codebase_for_id(tmp_path, "US-102")
    assert "src/a.py" in found
    assert "src/c.md" in found
    assert "src/b.py" not in found


def test_scan_codebase_word_boundary(tmp_path: Path) -> None:
    """US-1 must NOT match US-10."""
    (tmp_path / "x.py").write_text("# refers to US-10 only\n", encoding="utf-8")
    assert _scan_codebase_for_id(tmp_path, "US-1") == []


def test_scan_codebase_skips_noise_dirs(tmp_path: Path) -> None:
    """node_modules / .venv etc. must be ignored."""
    (tmp_path / "node_modules" / "junk").mkdir(parents=True)
    (tmp_path / "node_modules" / "junk" / "x.js").write_text("US-1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.js").write_text("US-1\n", encoding="utf-8")
    found = _scan_codebase_for_id(tmp_path, "US-1")
    assert found == ["src/x.js"]


def test_scan_codebase_ignores_non_text_extensions(tmp_path: Path) -> None:
    (tmp_path / "x.bin").write_bytes(b"US-1\n")
    (tmp_path / "x.py").write_text("US-1\n", encoding="utf-8")
    found = _scan_codebase_for_id(tmp_path, "US-1")
    assert found == ["x.py"]


# ---------------------------------------------------------------------------
# AuditEngine — needs a fake pipeline returning controlled metadata
# ---------------------------------------------------------------------------


class _FakePipeline:
    """RagPipeline stub that returns hits constructed from fixtures."""

    def __init__(self, hits_by_query: dict[str, list[Hit]] | None = None,
                 all_hits: list[Hit] | None = None) -> None:
        self._by_query = hits_by_query or {}
        self._all_hits = all_hits or []

    def ingest(self, documents) -> None:  # pragma: no cover
        pass

    def query(self, question: str, *, k: int = 5, where=None) -> QueryResult:
        if question == "*":
            return QueryResult(question=question, hits=self._all_hits[:k])
        return QueryResult(question=question, hits=self._by_query.get(question, [])[:k])


def _hit(spec_id: str, **meta: Any) -> Hit:
    return Hit(
        chunk_id=f"{spec_id}::chunk-0000",
        doc_id=spec_id,
        text="...",
        score=1.0,
        metadata={"kind": "story", "lifecycle": "implemented", **meta},
    )


def test_audit_spec_unknown_returns_none(tmp_path: Path) -> None:
    engine = AuditEngine(_FakePipeline(), code_root=tmp_path)
    assert engine.audit_spec("US-DOESNTEXIST") is None


def test_audit_spec_no_drift_when_files_exist(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "tasks.py"
    f.write_text("# US-102\ndef list_tasks(): ...\n", encoding="utf-8")

    pipeline = _FakePipeline(hits_by_query={
        "US-102": [_hit("US-102",
                       related_files="src/tasks.py",
                       template_compliance="full")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-102")
    assert report is not None
    assert report.severity == "none"
    assert not report.has_drift


def test_audit_status_drift_when_file_missing(tmp_path: Path) -> None:
    pipeline = _FakePipeline(hits_by_query={
        "US-102": [_hit("US-102",
                       related_files="src/nonexistent.py",
                       template_compliance="full")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-102")
    assert report.severity == "high"
    types = [i.type for i in report.issues]
    assert "status" in types


def test_audit_implemented_without_related_files_low_severity(tmp_path: Path) -> None:
    pipeline = _FakePipeline(hits_by_query={
        "US-X": [_hit("US-X", related_files="", template_compliance="full")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-X")
    assert report.severity == "low"
    assert any(i.type == "status" for i in report.issues)


def test_audit_coverage_over_claim(tmp_path: Path) -> None:
    """Declared file exists but doesn't reference the spec id."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tasks.py").write_text("def something(): pass\n", encoding="utf-8")

    pipeline = _FakePipeline(hits_by_query={
        "US-102": [_hit("US-102",
                       related_files="src/tasks.py",
                       template_compliance="full")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-102")
    types = [i.type for i in report.issues]
    assert "coverage_over" in types


def test_audit_coverage_under_claim(tmp_path: Path) -> None:
    """Code references the id but the spec doesn't declare the file."""
    (tmp_path / "src").mkdir()
    declared = tmp_path / "src" / "tasks.py"
    declared.write_text("# US-102\n", encoding="utf-8")
    extra = tmp_path / "src" / "helpers.py"
    extra.write_text("# also references US-102 indirectly\n", encoding="utf-8")

    pipeline = _FakePipeline(hits_by_query={
        "US-102": [_hit("US-102",
                       related_files="src/tasks.py",
                       template_compliance="full")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-102")
    types = [i.type for i in report.issues]
    assert "coverage_under" in types
    under_issue = next(i for i in report.issues if i.type == "coverage_under")
    assert "src/helpers.py" in under_issue.details["under_claimed"]


def test_audit_template_drift_surfaced_as_issue(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("# US-T\n", encoding="utf-8")
    pipeline = _FakePipeline(hits_by_query={
        "US-T": [_hit("US-T",
                     related_files="x.py",
                     template_compliance="partial",
                     kind="story",
                     lifecycle="proposed")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-T")
    types = [i.type for i in report.issues]
    assert "template" in types


def test_audit_template_full_no_template_issue(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("# US-Q\n", encoding="utf-8")
    pipeline = _FakePipeline(hits_by_query={
        "US-Q": [_hit("US-Q",
                     related_files="x.py",
                     template_compliance="full",
                     lifecycle="proposed")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-Q")
    assert not any(i.type == "template" for i in report.issues)


def test_audit_all_lifecycle_filter(tmp_path: Path) -> None:
    impl_hit = _hit("US-1", related_files="x.py", template_compliance="full",
                    lifecycle="implemented")
    prop_hit = _hit("US-2", related_files="", template_compliance="full",
                    lifecycle="proposed")
    pipeline = _FakePipeline(all_hits=[impl_hit, prop_hit])
    engine = AuditEngine(pipeline, code_root=tmp_path)

    impl_only = engine.audit_all(lifecycle="implemented")
    assert {r.spec_id for r in impl_only} == {"US-1"}

    proposed_only = engine.audit_all(lifecycle="proposed")
    assert {r.spec_id for r in proposed_only} == {"US-2"}


def test_audit_proposed_spec_skips_status_check(tmp_path: Path) -> None:
    """Status drift only applies to `implemented` specs."""
    pipeline = _FakePipeline(hits_by_query={
        "US-P": [_hit("US-P",
                     related_files="missing.py",
                     template_compliance="full",
                     lifecycle="proposed")]
    })
    engine = AuditEngine(pipeline, code_root=tmp_path)
    report = engine.audit_spec("US-P")
    assert not any(i.type == "status" for i in report.issues)


def test_drift_report_to_dict_round_trip(tmp_path: Path) -> None:
    report = DriftReport(
        spec_id="X", kind="story", lifecycle="implemented",
        template_compliance="full", severity="high",
        issues=[DriftIssue("status", "high", "d", "a")],
    )
    d = report.to_dict()
    assert d["spec_id"] == "X"
    assert d["has_drift"] is True
    assert d["issues"][0]["type"] == "status"
