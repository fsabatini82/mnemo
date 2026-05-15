"""Factory: turn an `--agentic <value>` flag into the right runner.

The user-facing flag accepts:

- `None` (flag absent)            → deterministic loader, no LLM
- `"copilot"`                     → subprocess `CopilotRunner` (auto-model)
- a family alias (`gpt`, `claude`, `sonnet`, `opus`)
                                  → resolve via settings.family_* and dispatch to GH Models
- a curated short-name (`gpt-5-mini`, `claude-opus-4-6`, …)
                                  → resolve to full id and dispatch to GH Models
- a full id (`openai/gpt-5`, `anthropic/claude-sonnet-4-6`)
                                  → pass-through to GH Models

This module is the single dispatch point — CLI commands (`mnemo-ingest`,
`mnemo-audit`) and the admin commands all go through it.
"""

from __future__ import annotations

import logging
from typing import Any

from mnemo.config import Settings
from mnemo.ingestion.agents.copilot.runner import CopilotRunner
from mnemo.ingestion.agents.gh_models.runner import GitHubModelsRunner
from mnemo.models_catalog import (
    ModelResolutionError,
    is_subprocess_marker,
    resolve_model,
)

logger = logging.getLogger(__name__)


class RunnerBuildError(RuntimeError):
    """Raised when the requested runner can't be constructed."""


def build_runner(
    agentic_value: str | None,
    settings: Settings,
    *,
    reasoning_effort: str | None = None,
) -> Any:
    """Construct the runner selected by `agentic_value`.

    Args:
        agentic_value: The raw value from `--agentic`. `None` means the
            caller does NOT want an LLM runner; this function then raises
            (the CLI layer should branch before calling this when no
            agentic mode is requested).
        settings: Active `Settings` instance (used for family resolution
            and GitHub Models defaults).
        reasoning_effort: Optional override of `settings.ghmodels_reasoning_effort`.
            Pass this from CLI flags or task-specific defaults
            (low for ingest, medium for audit).

    Returns:
        A runner instance whose surface is `is_available()`, `run()`,
        `run_json()`, `describe()`.
    """
    if agentic_value is None:
        raise RunnerBuildError(
            "build_runner() called without --agentic value; "
            "the deterministic path should not reach this function."
        )

    if is_subprocess_marker(agentic_value):
        return CopilotRunner()

    try:
        full_model = resolve_model(agentic_value, settings)
    except ModelResolutionError as exc:
        raise RunnerBuildError(str(exc)) from exc

    effective_effort = reasoning_effort or settings.ghmodels_reasoning_effort
    return GitHubModelsRunner(
        model=full_model,
        endpoint=settings.ghmodels_endpoint,
        reasoning_effort=effective_effort,
        max_completion_tokens=settings.ghmodels_max_completion_tokens,
        timeout=settings.ghmodels_timeout_seconds,
    )
