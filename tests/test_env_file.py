"""Tests for the .env atomic editor."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemo.env_file import (
    add_audit_comment,
    atomic_write,
    commit_change,
    find_value,
    read_env,
    update_or_append,
)


# ---------------------------------------------------------------------------
# read_env / find_value
# ---------------------------------------------------------------------------


def test_read_env_missing_returns_empty(tmp_path: Path) -> None:
    assert read_env(tmp_path / "no-such.env") == []


def test_read_env_returns_lines(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("# comment\nKEY=value\n\nOTHER=2\n", encoding="utf-8")
    assert read_env(p) == ["# comment", "KEY=value", "", "OTHER=2"]


def test_find_value_present() -> None:
    lines = ["# preamble", "FOO=bar", "BAZ=qux"]
    assert find_value(lines, "FOO") == "bar"
    assert find_value(lines, "BAZ") == "qux"


def test_find_value_absent() -> None:
    assert find_value(["FOO=1"], "BAR") is None


def test_find_value_ignores_lines_that_are_not_kv() -> None:
    """Comments and blanks must not be matched as KEY=value."""
    assert find_value(["# FOO=fake", "real-comment", ""], "FOO") is None


# ---------------------------------------------------------------------------
# update_or_append
# ---------------------------------------------------------------------------


def test_update_existing_returns_old_value() -> None:
    lines = ["KEY=old"]
    new_lines, old = update_or_append(lines, "KEY", "new")
    assert old == "old"
    assert new_lines == ["KEY=new"]


def test_update_preserves_surrounding_content() -> None:
    lines = [
        "# preamble",
        "FIRST=1",
        "",
        "KEY=old",
        "",
        "LAST=z",
    ]
    new_lines, old = update_or_append(lines, "KEY", "new")
    assert old == "old"
    assert new_lines == ["# preamble", "FIRST=1", "", "KEY=new", "", "LAST=z"]


def test_append_when_key_absent_returns_none() -> None:
    lines = ["FIRST=1"]
    new_lines, old = update_or_append(lines, "KEY", "new")
    assert old is None
    assert new_lines[-1] == "KEY=new"


def test_append_adds_separator_when_last_line_not_empty() -> None:
    lines = ["FIRST=1"]
    new_lines, _ = update_or_append(lines, "NEW", "x")
    # Empty line separator before the new key.
    assert new_lines == ["FIRST=1", "", "NEW=x"]


def test_append_to_empty_file() -> None:
    new_lines, old = update_or_append([], "FOO", "bar")
    assert old is None
    assert new_lines == ["FOO=bar"]


def test_update_only_replaces_first_match() -> None:
    """Duplicate keys: only the first occurrence is replaced."""
    lines = ["KEY=v1", "KEY=v2"]
    new_lines, old = update_or_append(lines, "KEY", "v3")
    assert old == "v1"
    assert new_lines == ["KEY=v3", "KEY=v2"]


# ---------------------------------------------------------------------------
# add_audit_comment
# ---------------------------------------------------------------------------


def test_audit_comment_inserted_above_key_line() -> None:
    lines = ["FIRST=1", "KEY=value", "LAST=z"]
    out = add_audit_comment(lines, "KEY", old_value="old")
    # The comment is inserted before the KEY line, others preserved.
    assert out[0] == "FIRST=1"
    assert out[1].startswith("# changed via MCP at ")
    assert "was: old" in out[1]
    assert out[2] == "KEY=value"
    assert out[3] == "LAST=z"


def test_audit_comment_for_new_key_records_unset() -> None:
    lines = ["KEY=value"]
    out = add_audit_comment(lines, "KEY", old_value=None)
    assert "<unset>" in out[0]


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------


def test_atomic_write_creates_file_with_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "out.env"
    atomic_write(p, ["FOO=1", "BAR=2"])
    text = p.read_text(encoding="utf-8")
    assert text == "FOO=1\nBAR=2\n"


def test_atomic_write_no_tmp_left_over(tmp_path: Path) -> None:
    p = tmp_path / "out.env"
    atomic_write(p, ["X=1"])
    assert p.is_file()
    assert not (tmp_path / "out.env.tmp").exists()


def test_atomic_write_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "dir" / "out.env"
    atomic_write(p, ["X=1"])
    assert p.is_file()


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    p = tmp_path / "out.env"
    p.write_text("OLD=stuff\n", encoding="utf-8")
    atomic_write(p, ["NEW=stuff"])
    assert "OLD" not in p.read_text(encoding="utf-8")
    assert "NEW=stuff" in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# commit_change (end-to-end)
# ---------------------------------------------------------------------------


def test_commit_change_existing_key(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("# header\nFOO=old\nBAR=other\n", encoding="utf-8")
    old = commit_change(p, "FOO", "new")
    assert old == "old"
    text = p.read_text(encoding="utf-8")
    assert "FOO=new" in text
    assert "FOO=old" not in text
    # Audit comment was inserted.
    assert "# changed via MCP at " in text
    assert "(was: old)" in text
    # Other content preserved.
    assert "BAR=other" in text
    assert text.splitlines()[0] == "# header"


def test_commit_change_new_key(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=1\n", encoding="utf-8")
    old = commit_change(p, "NEW", "value")
    assert old is None
    text = p.read_text(encoding="utf-8")
    assert "NEW=value" in text
    assert "(was: <unset>)" in text
    # Original preserved.
    assert "FOO=1" in text


def test_commit_change_creates_file_if_absent(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    old = commit_change(p, "KEY", "value")
    assert old is None
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "KEY=value" in text
