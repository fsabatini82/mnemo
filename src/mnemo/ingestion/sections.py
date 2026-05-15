"""Markdown section parsing utilities.

The specs loader uses these to extract canonical template sections
(`## User Story`, `## Acceptance Criteria`, `### Happy path`, …) from
the body of a spec file. The extracted sections become flat metadata
fields on the resulting `Document`, so an IDE agent can request just
the slice it needs.

Heading matching is **case-insensitive** and tolerates trailing
whitespace, but otherwise looks for an exact textual match (no fuzzy
matching). If a section is missing or named differently, it's
silently absent from the result — the loader falls back to empty
metadata for that field.
"""

from __future__ import annotations

import re

# H2 splitter: matches lines that begin with `## ` at the start of a line.
_H2_SPLIT = re.compile(r"(?m)^##\s+(.+?)\s*$")
# H3 splitter: same shape for one level deeper.
_H3_SPLIT = re.compile(r"(?m)^###\s+(.+?)\s*$")


def extract_h2_sections(body: str) -> dict[str, str]:
    """Split a markdown body into `{lower-cased heading: section body}`.

    The text before the first H2 is discarded (it's the document's
    preamble below the H1 title). Section bodies are stripped of
    trailing whitespace.
    """
    return _extract(body, _H2_SPLIT)


def extract_h3_sections(body: str) -> dict[str, str]:
    """Same as `extract_h2_sections` but for H3-level headings.

    Useful for parsing nested structures (e.g. `### Happy path` inside
    `## Test Scenarios`).
    """
    return _extract(body, _H3_SPLIT)


def _extract(body: str, splitter: re.Pattern[str]) -> dict[str, str]:
    matches = list(splitter.finditer(body))
    if not matches:
        return {}

    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_body = body[start:end].strip()
        result[heading] = section_body
    return result
