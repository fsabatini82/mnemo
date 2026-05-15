"""Tests for markdown section parsing helpers."""

from __future__ import annotations

from mnemo.ingestion.sections import extract_h2_sections, extract_h3_sections


def test_extract_h2_simple() -> None:
    body = "## A\nbody of A\n\n## B\nbody of B\n"
    out = extract_h2_sections(body)
    assert out == {"a": "body of A", "b": "body of B"}


def test_extract_h2_lowercases_headings() -> None:
    body = "## User Story\nbody\n"
    assert "user story" in extract_h2_sections(body)


def test_extract_h2_ignores_preamble() -> None:
    body = "Preamble text\n\n## First\nfirst body\n"
    assert extract_h2_sections(body) == {"first": "first body"}


def test_extract_h2_no_headings_returns_empty() -> None:
    assert extract_h2_sections("just a paragraph") == {}


def test_extract_h2_strips_trailing_whitespace() -> None:
    body = "## Title\n\n  body  \n\n\n"
    assert extract_h2_sections(body)["title"] == "body"


def test_extract_h2_preserves_inner_h3() -> None:
    body = "## Test Scenarios\n\n### Happy\ngherkin happy\n\n### Edge\ngherkin edge\n"
    out = extract_h2_sections(body)
    section = out["test scenarios"]
    assert "### Happy" in section
    assert "### Edge" in section


def test_extract_h3_inside_section() -> None:
    body = "### Happy path\ngherkin1\n\n### Error path\ngherkin2\n"
    out = extract_h3_sections(body)
    assert out == {"happy path": "gherkin1", "error path": "gherkin2"}


def test_h3_ignores_h2_headers() -> None:
    body = "## H2\nsome text\n### H3\ndeeper\n"
    # extract_h3 should pick only ### lines.
    out = extract_h3_sections(body)
    assert out == {"h3": "deeper"}
