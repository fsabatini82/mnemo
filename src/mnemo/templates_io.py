"""Spec template loading + scaffolding helpers.

Templates ship with the package under `mnemo/templates/<kind>.template.md`
and define the canonical section structure for each spec kind:

- **story** — User Story + Acceptance Criteria + Test Scenarios (Happy /
  Error / Edge) + Acceptance Summary.
- **adr**   — Context + Decision + Consequences + Acceptance Summary.
- **epic**  — Goals + Constraints + Stories in scope + Acceptance Summary.

The deterministic loader and the Copilot agent both extract these
sections into flat metadata fields so an IDE agent can query just the
piece it needs (e.g. "give me the acceptance criteria of US-102").

The `Acceptance Summary` is the load-bearing field for drift detection
in F4 — keep it tight, in the team's own words, and update it when the
spec changes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

TemplateKind = Literal["story", "adr", "epic"]
TEMPLATE_KINDS: tuple[TemplateKind, ...] = ("story", "adr", "epic")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class TemplateError(RuntimeError):
    """Raised when a template can't be found or rendered."""


def list_kinds() -> tuple[TemplateKind, ...]:
    """Return the canonical template kinds shipped with Mnemo."""
    return TEMPLATE_KINDS


def template_path(kind: TemplateKind) -> Path:
    """Return the absolute path to the template file for `kind`."""
    if kind not in TEMPLATE_KINDS:
        raise TemplateError(
            f"Unknown template kind {kind!r}. Available: {', '.join(TEMPLATE_KINDS)}"
        )
    path = _TEMPLATES_DIR / f"{kind}.template.md"
    if not path.is_file():
        raise TemplateError(
            f"Template file for kind {kind!r} not found at {path}. "
            "The package may be installed incorrectly — check the wheel."
        )
    return path


def load_template(kind: TemplateKind) -> str:
    """Return the raw template content as a string."""
    return template_path(kind).read_text(encoding="utf-8")


def render(kind: TemplateKind, **vars: str) -> str:
    """Render a template by substituting `{{var}}` placeholders.

    Unknown placeholders are kept as-is in the output (so a user
    scaffolding a story without `--description` still sees
    `{{description}}` and knows to fill it in).
    """
    content = load_template(kind)

    def _sub(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return str(vars.get(var_name, match.group(0)))

    return _PLACEHOLDER_RE.sub(_sub, content)
