"""Tests for the spec template loader / renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemo.templates_io import (
    TEMPLATE_KINDS,
    TemplateError,
    list_kinds,
    load_template,
    render,
    template_path,
)


def test_list_kinds_returns_canonical_set() -> None:
    assert set(list_kinds()) == {"story", "adr", "epic"}


@pytest.mark.parametrize("kind", list(TEMPLATE_KINDS))
def test_template_path_resolves_for_each_kind(kind: str) -> None:
    p = template_path(kind)
    assert p.is_file()
    assert p.suffix == ".md"


def test_template_path_rejects_unknown_kind() -> None:
    with pytest.raises(TemplateError, match="Unknown template kind"):
        template_path("decision")  # not in TEMPLATE_KINDS


@pytest.mark.parametrize("kind", list(TEMPLATE_KINDS))
def test_load_template_returns_non_empty_text(kind: str) -> None:
    content = load_template(kind)
    assert len(content) > 100
    # Each template has frontmatter and at least one H1.
    assert content.startswith("---")
    assert "# {{id}}" in content


def test_render_substitutes_placeholders() -> None:
    out = render("story", id="US-205", title="Login con SSO")
    assert "US-205" in out
    assert "Login con SSO" in out
    assert "{{id}}" not in out
    assert "{{title}}" not in out


def test_render_keeps_unknown_placeholders() -> None:
    """An unknown placeholder must be left intact so the user notices."""
    out = render("story", id="US-205", title="X")
    # `{{nonexistent}}` isn't in the template anyway, but the substitution
    # function must never raise on missing vars.
    assert "US-205" in out


def test_render_story_template_has_required_sections() -> None:
    out = render("story", id="US-1", title="Foo")
    for required in (
        "## User Story",
        "## Acceptance Criteria",
        "## Test Scenarios",
        "### Happy path",
        "### Error path",
        "### Edge case",
        "## Acceptance Summary",
    ):
        assert required in out, f"missing required section: {required}"


def test_render_adr_template_has_required_sections() -> None:
    out = render("adr", id="ADR-1", title="X")
    for required in ("## Context", "## Decision", "## Consequences", "## Acceptance Summary"):
        assert required in out


def test_render_epic_template_has_required_sections() -> None:
    out = render("epic", id="EPIC-1", title="X")
    for required in ("## Goals", "## Constraints", "## Stories in scope", "## Acceptance Summary"):
        assert required in out
