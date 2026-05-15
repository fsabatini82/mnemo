"""Tests for the specs ingestion adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

import datetime as _dt

from mnemo.ingestion.specs_loader import (
    _flatten_metadata,
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


def test_flatten_metadata_handles_dates() -> None:
    # YAML auto-converts `2025-06-14` into datetime.date; Chroma 1.x rejects it.
    flat = _flatten_metadata({"date": _dt.date(2025, 6, 14)})
    assert flat == {"date": "2025-06-14"}


def test_flatten_metadata_joins_lists() -> None:
    flat = _flatten_metadata({"related_bugs": ["BUG-1", "BUG-2"]})
    assert flat == {"related_bugs": "BUG-1, BUG-2"}


def test_flatten_metadata_preserves_scalars() -> None:
    src = {"a": "x", "b": 1, "c": 1.5, "d": True, "e": None}
    assert _flatten_metadata(src) == src


def test_flatten_metadata_stringifies_dicts() -> None:
    flat = _flatten_metadata({"obj": {"k": "v"}})
    assert flat["obj"] == "{'k': 'v'}"


def test_parse_spec_flattens_yaml_date_field(tmp_path: Path) -> None:
    """Regression for the production failure on ADR-002 (YAML `date:` field)."""
    p = tmp_path / "adrs" / "ADR-X.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: ADR-X\ntitle: Test\ndate: 2025-06-14\n---\n\nBody.\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    assert isinstance(doc.metadata["date"], str)
    assert doc.metadata["date"] == "2025-06-14"


def test_load_specs_returns_all_documents(specs_dir: Path) -> None:
    docs = load_specs(specs_dir)
    assert len(docs) == 4
    ids = {d.id for d in docs}
    assert "US-001" in ids
    assert "EPIC-BE" in ids
    assert "ADR-001" in ids


# ---------------------------------------------------------------------------
# Lifecycle field (F2)
# ---------------------------------------------------------------------------


def test_parse_spec_normalizes_canonical_lifecycle(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-L1.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: US-L1\ntitle: T\nlifecycle: IN_PROGRESS\n---\n\nBody.\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["lifecycle"] == "in-progress"


def test_parse_spec_lifecycle_absent_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-L2.md"
    p.parent.mkdir(parents=True)
    p.write_text("---\nid: US-L2\ntitle: T\n---\n\nBody.\n", encoding="utf-8")
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["lifecycle"] == ""


def test_parse_spec_lifecycle_non_canonical_stored_as_is(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-L3.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: US-L3\ntitle: T\nlifecycle: draft\n---\n\nBody.\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    # Non-canonical values are still stored — they just won't match strict filters.
    assert doc.metadata["lifecycle"] == "draft"


# ---------------------------------------------------------------------------
# Template sections + compliance (F3)
# ---------------------------------------------------------------------------


_FULL_STORY = """---
id: US-T1
title: Full template story
kind: story
lifecycle: proposed
---

# US-T1 — Full template story

## User Story
Come dev voglio testare così da validare.

## Acceptance Criteria
- AC1: passa il test
- AC2: niente regressioni

## Test Scenarios

### Happy path
```gherkin
Given setup
When azione
Then ok
```

### Error path
```gherkin
Given setup
When errore
Then 500
```

### Edge case
```gherkin
Given limite
When azione
Then comportamento boundary
```

## Acceptance Summary

Story con tutte le sezioni canoniche, deve essere classificata full.
"""


def test_parse_spec_extracts_all_story_sections(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-T1.md"
    p.parent.mkdir(parents=True)
    p.write_text(_FULL_STORY, encoding="utf-8")
    doc = parse_spec(p, tmp_path)
    assert "voglio testare" in doc.metadata["user_story"]
    assert "AC1" in doc.metadata["acceptance_criteria"]
    assert "Given setup" in doc.metadata["test_scenarios_happy"]
    assert "500" in doc.metadata["test_scenarios_error"]
    assert "limite" in doc.metadata["test_scenarios_edge"]
    assert "tutte le sezioni" in doc.metadata["acceptance_summary"]
    assert doc.metadata["template_compliance"] == "full"


def test_parse_spec_missing_sections_classified_partial(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-T2.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: US-T2\ntitle: T\nkind: story\n---\n\n"
        "# US-T2\n\n## Acceptance Criteria\n- only AC, nothing else\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["template_compliance"] == "partial"
    assert doc.metadata["acceptance_criteria"]
    assert doc.metadata["user_story"] == ""


def test_parse_spec_no_sections_classified_non_compliant(tmp_path: Path) -> None:
    p = tmp_path / "stories" / "US-T3.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: US-T3\ntitle: T\nkind: story\n---\n\n# US-T3\n\nJust prose.\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["template_compliance"] == "non-compliant"


def test_parse_spec_unknown_kind_classified_na(tmp_path: Path) -> None:
    p = tmp_path / "other" / "X.md"
    p.parent.mkdir(parents=True)
    p.write_text("---\nid: X\nkind: glossary\n---\n\nBody\n", encoding="utf-8")
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["template_compliance"] == "n/a"


def test_parse_spec_adr_full_compliance(tmp_path: Path) -> None:
    p = tmp_path / "adrs" / "ADR-T.md"
    p.parent.mkdir(parents=True)
    p.write_text(
        "---\nid: ADR-T\ntitle: T\nkind: adr\n---\n\n# ADR-T\n\n"
        "## Context\nwhy\n\n## Decision\ndo X\n\n## Consequences\npro/contro\n",
        encoding="utf-8",
    )
    doc = parse_spec(p, tmp_path)
    assert doc.metadata["template_compliance"] == "full"
    assert doc.metadata["context"] == "why"
    assert doc.metadata["decision"] == "do X"
    assert doc.metadata["consequences"] == "pro/contro"
