"""Canonical lifecycle vocabulary for spec records.

A spec's `lifecycle` field tracks where it sits in the product timeline:

- `proposed`      — drafted, not yet picked up for implementation
- `in-progress`   — actively being implemented
- `implemented`   — shipped and present in the codebase
- `superseded`    — previously implemented but obsoleted by a newer decision
- `as-is`         — reverse-engineering notes / documentation of legacy code

Ingestion is **permissive** about unknown values (we log a warning and
store the raw string), so legacy spec corpora don't break. Filtering
is **strict** — only canonical values are accepted in the MCP tool's
filter parameter, to prevent silent mismatches caused by typos.
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

logger = logging.getLogger(__name__)

LifecycleValue = Literal[
    "proposed",
    "in-progress",
    "implemented",
    "superseded",
    "as-is",
]

LIFECYCLE_VALUES: tuple[LifecycleValue, ...] = get_args(LifecycleValue)


def normalize(raw: object | None) -> str:
    """Normalize a raw frontmatter value into the canonical form.

    - None / "" → "" (treated as "unspecified" downstream)
    - Strings are trimmed, lowercased, and `_` is folded to `-`.
    - Values outside the canonical set are passed through with a
      warning, so the user notices but ingestion continues.
    """
    if raw is None:
        return ""
    text = str(raw).strip().lower().replace("_", "-")
    if not text:
        return ""
    if text not in LIFECYCLE_VALUES:
        logger.warning(
            "Non-canonical lifecycle value %r; accepted: %s. "
            "Storing as-is; this value won't match strict filters.",
            text, ", ".join(LIFECYCLE_VALUES),
        )
    return text


def is_canonical(value: str) -> bool:
    return value in LIFECYCLE_VALUES


def assert_canonical(value: str) -> str:
    """Raise ValueError if `value` is not in the canonical set.

    Used at filter time (strict path): the MCP tool, `mnemo-admin`
    queries, etc.
    """
    if value not in LIFECYCLE_VALUES:
        raise ValueError(
            f"Invalid lifecycle filter {value!r}: must be one of "
            f"{', '.join(LIFECYCLE_VALUES)}."
        )
    return value
