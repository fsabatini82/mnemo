"""Tests for the lifecycle vocabulary module."""

from __future__ import annotations

import logging

import pytest

from mnemo.core.protocols import matches_where
from mnemo.lifecycle import (
    LIFECYCLE_VALUES,
    assert_canonical,
    is_canonical,
    normalize,
)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_none_returns_empty() -> None:
    assert normalize(None) == ""


def test_normalize_empty_string_returns_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("proposed", "proposed"),
        ("PROPOSED", "proposed"),
        ("  Proposed  ", "proposed"),
        ("in-progress", "in-progress"),
        ("in_progress", "in-progress"),  # underscore folded to dash
        ("IN-PROGRESS", "in-progress"),
        ("implemented", "implemented"),
        ("superseded", "superseded"),
        ("as-is", "as-is"),
        ("as_is", "as-is"),
    ],
)
def test_normalize_canonical_values(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_non_canonical_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="mnemo.lifecycle"):
        result = normalize("draft")  # not canonical
    assert result == "draft"  # stored as-is, not silently coerced
    assert any("Non-canonical lifecycle" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# is_canonical() / assert_canonical()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", list(LIFECYCLE_VALUES))
def test_is_canonical_true_for_each_enum_value(value: str) -> None:
    assert is_canonical(value)


@pytest.mark.parametrize("value", ["draft", "Implemented", "todo", "", "in_progress"])
def test_is_canonical_false_for_others(value: str) -> None:
    assert not is_canonical(value)


def test_assert_canonical_passes_for_valid() -> None:
    assert assert_canonical("implemented") == "implemented"


def test_assert_canonical_raises_for_invalid() -> None:
    with pytest.raises(ValueError, match="lifecycle filter"):
        assert_canonical("draft")


# ---------------------------------------------------------------------------
# matches_where() shared helper
# ---------------------------------------------------------------------------


def test_matches_where_simple_equality() -> None:
    md = {"lifecycle": "implemented", "kind": "story"}
    assert matches_where(md, {"lifecycle": "implemented"})
    assert not matches_where(md, {"lifecycle": "proposed"})


def test_matches_where_multiple_keys_all_must_match() -> None:
    md = {"lifecycle": "implemented", "kind": "story"}
    assert matches_where(md, {"lifecycle": "implemented", "kind": "story"})
    assert not matches_where(md, {"lifecycle": "implemented", "kind": "adr"})


def test_matches_where_in_operator() -> None:
    md = {"lifecycle": "proposed"}
    assert matches_where(md, {"lifecycle": {"$in": ["proposed", "in-progress"]}})
    assert not matches_where(md, {"lifecycle": {"$in": ["implemented"]}})


def test_matches_where_ne_operator() -> None:
    md = {"lifecycle": "implemented"}
    assert matches_where(md, {"lifecycle": {"$ne": "proposed"}})
    assert not matches_where(md, {"lifecycle": {"$ne": "implemented"}})


def test_matches_where_missing_key_fails() -> None:
    assert not matches_where({}, {"lifecycle": "implemented"})


def test_matches_where_unknown_operator_safe_reject() -> None:
    """An unrecognized operator must not produce a false positive."""
    md = {"lifecycle": "implemented"}
    assert not matches_where(md, {"lifecycle": {"$weird": "implemented"}})
