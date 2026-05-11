"""Tests for the bugs ingestion adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemo.ingestion.bugs_loader import (
    _render_indexed_text,
    iter_bug_files,
    load_bugs,
    parse_bug,
)


def test_iter_bug_files_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_bug_files(tmp_path / "missing"))


def test_iter_bug_files_yields_json(bugs_dir: Path) -> None:
    files = list(iter_bug_files(bugs_dir))
    assert len(files) == 2
    assert all(f.suffix == ".json" for f in files)


def test_parse_bug_full_record(bugs_dir: Path) -> None:
    path = bugs_dir / "BUG-001.json"
    doc = parse_bug(path)
    assert doc.id == "BUG-001"
    assert doc.metadata["kind"] == "bug"
    assert doc.metadata["severity"] == "high"
    assert doc.metadata["related_spec"] == "US-001"
    # Lists are flattened to comma-separated strings for vector-store compat.
    assert "a.py" in doc.metadata["files_touched"]
    assert "typo" in doc.metadata["pattern_tags"]
    # Symptom + root cause + fix landed in the indexed text.
    assert "Something fails" in doc.text
    assert "Fix the typo" in doc.text


def test_parse_bug_minimal_record(bugs_dir: Path) -> None:
    """A bug record with only id+title should still parse cleanly."""
    path = bugs_dir / "BUG-002.json"
    doc = parse_bug(path)
    assert doc.id == "BUG-002"
    assert doc.metadata["title"] == "Minimal record"
    # Missing fields default to empty strings.
    assert doc.metadata["severity"] == ""
    assert doc.metadata["files_touched"] == ""


def test_parse_bug_falls_back_to_filename_when_no_id(tmp_path: Path) -> None:
    p = tmp_path / "BUG-orphan.json"
    p.write_text(json.dumps({"title": "No id"}), encoding="utf-8")
    doc = parse_bug(p)
    assert doc.id == "BUG-orphan"  # falls back to filename stem


def test_render_indexed_text_includes_title_and_sections() -> None:
    record = {
        "title": "Title",
        "symptom": "It breaks",
        "root_cause": "Because Y",
        "fix_summary": "Do Z",
        "files_touched": ["a.py", "b.py"],
        "pattern_tags": ["regression"],
    }
    out = _render_indexed_text("BUG-X", record)
    assert "BUG-X" in out
    assert "Title" in out
    assert "It breaks" in out
    assert "Because Y" in out
    assert "Do Z" in out
    assert "a.py" in out
    assert "regression" in out


def test_render_indexed_text_minimal_record() -> None:
    out = _render_indexed_text("BUG-Y", {"title": "Just a title"})
    assert "BUG-Y" in out
    assert "Just a title" in out


def test_load_bugs_returns_all(bugs_dir: Path) -> None:
    docs = load_bugs(bugs_dir)
    assert len(docs) == 2
    assert {d.id for d in docs} == {"BUG-001", "BUG-002"}
