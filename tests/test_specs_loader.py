"""Tests for the specs ingestion adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemo.ingestion.specs_loader import (
    _infer_kind,
    _render_indexed_text,
    _split_frontmatter,
    iter_spec_files,
    load_specs,
    parse_spec,
)


def test_iter_spec_files_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_spec_files(tmp_path / "nonexistent"))


def test_iter_spec_files_yields_markdown(specs_dir: Path) -> None:
    files = list(iter_spec_files(specs_dir))
    assert len(files) == 4
    assert all(f.suffix == ".md" for f in files)


def test_split_frontmatter_extracts_metadata() -> None:
    raw = "---\nid: X\ntitle: T\n---\n\nBody here.\n"
    meta, body = _split_frontmatter(raw)
    assert meta == {"id": "X", "title": "T"}
    assert "Body here" in body


def test_split_frontmatter_no_frontmatter_passthrough() -> None:
    raw = "# Plain markdown\n\nNo frontmatter.\n"
    meta, body = _split_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_split_frontmatter_with_non_dict_metadata_falls_back() -> None:
    # Frontmatter that parses to a list, not a dict — should be treated
    # as if there were no frontmatter at all.
    raw = "---\n- one\n- two\n---\n\nBody\n"
    meta, body = _split_frontmatter(raw)
    assert meta == {}
    assert body == raw


@pytest.mark.parametrize(
    "rel,expected",
    [
        ("epics/EPIC-X.md", "epic"),
        ("stories/US-1.md", "story"),
        ("adrs/ADR-1.md", "adr"),
        ("other.md", "spec"),
    ],
)
def test_infer_kind(tmp_path: Path, rel: str, expected: str) -> None:
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("body", encoding="utf-8")
    assert _infer_kind(full, tmp_path) == expected


def test_parse_spec_with_frontmatter(specs_dir: Path) -> None:
    story = specs_dir / "stories" / "US-001.md"
    doc = parse_spec(story, specs_dir)
    assert doc.id == "US-001"
    assert doc.metadata["title"] == "Test story"
    assert doc.metadata["kind"] == "story"
    assert "Test story" in doc.text  # Title rendered into indexed text.


def test_parse_spec_without_frontmatter_uses_relpath_as_id(specs_dir: Path) -> None:
    raw = specs_dir / "stories" / "raw.md"
    doc = parse_spec(raw, specs_dir)
    # ID falls back to the relative path when frontmatter is missing.
    assert "raw" in doc.id
    assert doc.metadata["kind"] == "story"


def test_render_indexed_text_includes_title_and_links() -> None:
    meta = {
        "id": "US-1",
        "title": "Title",
        "epic": "EPIC-X",
        "related_bugs": ["BUG-1", "BUG-2"],
        "related_adrs": ["ADR-1"],
    }
    rendered = _render_indexed_text(meta, "body")
    assert "US-1 — Title" in rendered
    assert "Epic: EPIC-X" in rendered
    assert "BUG-1, BUG-2" in rendered
    assert "ADR-1" in rendered
    assert rendered.endswith("body")


def test_render_indexed_text_minimal() -> None:
    rendered = _render_indexed_text({}, "just body")
    assert rendered == "just body"


def test_load_specs_returns_all_documents(specs_dir: Path) -> None:
    docs = load_specs(specs_dir)
    assert len(docs) == 4
    ids = {d.id for d in docs}
    assert "US-001" in ids
    assert "EPIC-BE" in ids
    assert "ADR-001" in ids
