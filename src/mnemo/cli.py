"""Mnemo CLI — ingestion entry point.

Subcommands feed the knowledge axes for the configured (project, env).
The first ingest for a new project auto-registers it in the project
registry (`<persist_dir>/projects.json`). Server-side reads stay
read-only and never touch the registry.

Three ingestion modes, opt-in per invocation:

- **deterministic** (no `--agentic` flag): the loader adapters parse
  files mechanically. Fast, predictable, no external dependencies.
- **agentic via GitHub Models** (`--agentic` or `--agentic <model>`):
  HTTP call to GH Models with the chosen model. Predictable costs,
  reasoning_effort tunable, requires a token with `models:read` scope.
- **agentic via subprocess Copilot CLI** (`--agentic copilot`): drives
  the local `gh copilot` / `copilot` binary in auto-model mode. Use
  when you don't want explicit model control and have the CLI
  installed locally.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from mnemo.config import load_settings
from mnemo.core.models import Document
from mnemo.core.protocols import RagPipeline
from mnemo.factory import build_system
from mnemo.ingestion.agents.runner_factory import RunnerBuildError, build_runner
from mnemo.ingestion.bugs_loader import load_bugs
from mnemo.ingestion.specs_loader import load_specs
from mnemo.models_catalog import list_shortnames
from mnemo.registry import ENVIRONMENTS, validate_environment, validate_slug

logger = logging.getLogger("mnemo")

# Task-specific reasoning_effort defaults: ingestion is high-volume structured
# extraction (low effort is fine); audit needs reasoning (medium).
_INGEST_DEFAULT_EFFORT = "low"


# ---------------------------------------------------------------------------
# Mode-specific loaders
# ---------------------------------------------------------------------------


def _build_agentic_runner(args: argparse.Namespace, settings: Any) -> Any | None:
    """Shared runner construction for both ingestion subcommands.

    Returns None for deterministic mode. On failure (unknown model,
    missing token/binary) exits with a clear message — does not fall
    back silently to deterministic.
    """
    agentic_value = getattr(args, "agentic", None)
    if agentic_value is None:
        return None

    effort_override = getattr(args, "reasoning_effort", None) or _INGEST_DEFAULT_EFFORT
    try:
        runner = build_runner(agentic_value, settings, reasoning_effort=effort_override)
    except RunnerBuildError as exc:
        logger.error("%s", exc)
        raise SystemExit(2)

    if not runner.is_available():
        logger.error(
            "Selected runner not ready: %s. "
            "Configure the required token/binary or omit --agentic to use "
            "the deterministic loader.",
            runner.describe(),
        )
        raise SystemExit(2)
    logger.info("Using agentic ingestion: %s", runner.describe())
    return runner


def _load_specs(source: Path, runner: Any | None) -> list[Document]:
    if runner is None:
        return load_specs(source)
    from mnemo.ingestion.agents.copilot.specs_agent import CopilotSpecsAgent

    return CopilotSpecsAgent(runner=runner).ingest(source)


def _load_bugs(source: Path, runner: Any | None) -> list[Document]:
    if runner is None:
        return load_bugs(source)
    from mnemo.ingestion.agents.copilot.bugs_agent import CopilotBugsAgent

    return CopilotBugsAgent(runner=runner).ingest(source)


# ---------------------------------------------------------------------------
# Subcommand entry points
# ---------------------------------------------------------------------------


def _ingest(pipeline: RagPipeline, documents: Sequence[Document], label: str) -> int:
    if not documents:
        logger.warning("No %s documents to ingest.", label)
        return 1
    logger.info("Ingesting %d %s document(s)...", len(documents), label)
    pipeline.ingest(documents)
    logger.info("Done — %s collection refreshed.", label)
    return 0


def _apply_overrides(args: argparse.Namespace) -> None:
    """Inject CLI overrides into the environment so Settings picks them up."""
    if getattr(args, "project", None):
        os.environ["MNEMO_PROJECT"] = args.project
    if getattr(args, "environment", None):
        os.environ["MNEMO_ENVIRONMENT"] = args.environment


def _cmd_specs(args: argparse.Namespace) -> int:
    _apply_overrides(args)
    settings = load_settings()
    source: Path = args.path or settings.specs_source_dir
    system = build_system(settings, auto_register=True)
    logger.info(
        "Target: project=%s (id=%s) env=%s collection=%s",
        settings.project, system.project_id, system.environment,
        f"{system.effective_prefix}_{settings.specs_collection}",
    )
    runner = _build_agentic_runner(args, settings)
    docs = _load_specs(source, runner)
    return _ingest(system.specs, docs, "specs")


def _cmd_bugs(args: argparse.Namespace) -> int:
    _apply_overrides(args)
    settings = load_settings()
    source: Path = args.path or settings.bugs_source_dir
    system = build_system(settings, auto_register=True)
    logger.info(
        "Target: project=%s (id=%s) env=%s collection=%s",
        settings.project, system.project_id, system.environment,
        f"{system.effective_prefix}_{settings.bugs_collection}",
    )
    runner = _build_agentic_runner(args, settings)
    docs = _load_bugs(source, runner)
    return _ingest(system.bugs, docs, "bugs")


def _cmd_all(args: argparse.Namespace) -> int:
    rc = _cmd_specs(args)
    rc |= _cmd_bugs(args)
    return rc


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _add_common_flags(parser: argparse.ArgumentParser, *, hide_path: bool = False) -> None:
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS if hide_path else "Override the source dir for this run.",
    )
    parser.add_argument(
        "--project",
        type=_slug,
        default=None,
        help="Project slug (lowercase, [a-z0-9-], max 32 chars). Default: from env/.env "
             "(or 'demo-project').",
    )
    parser.add_argument(
        "--env",
        dest="environment",
        choices=list(ENVIRONMENTS),
        default=None,
        help="Environment for this project. Default: from env/.env (or 'dev').",
    )
    parser.add_argument(
        "--agentic",
        nargs="?",
        const="gpt-5-mini",
        default=None,
        metavar="MODEL",
        help=(
            "Use an LLM-powered ingestion agent. Without a value: uses "
            "GitHub Models with gpt-5-mini. With `copilot`: uses the local "
            "subprocess Copilot CLI in auto-model mode. With a model name "
            "or full id: uses GitHub Models with that model. "
            f"Supported short-names: {', '.join(list_shortnames())}. "
            "Family aliases: gpt, claude, sonnet, opus (resolved via "
            "MNEMO_FAMILY_* settings). Or pass a full id (e.g. openai/gpt-5)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help=(
            "Override reasoning_effort for this run. "
            f"Default for ingestion: {_INGEST_DEFAULT_EFFORT} "
            "(structured extraction benefits little from heavier reasoning)."
        ),
    )


def _slug(value: str) -> str:
    """argparse type: validate slug, raise ArgumentTypeError on failure."""
    try:
        return validate_slug(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _env(value: str) -> str:
    try:
        return validate_environment(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemo-ingest", description="Mnemo ingestion CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_specs = sub.add_parser("specs", help="Ingest specs from a folder mirror.")
    _add_common_flags(p_specs)
    p_specs.set_defaults(func=_cmd_specs)

    p_bugs = sub.add_parser("bugs", help="Ingest resolved bugs from a folder mirror.")
    _add_common_flags(p_bugs)
    p_bugs.set_defaults(func=_cmd_bugs)

    p_all = sub.add_parser("all", help="Ingest both axes (specs + bugs).")
    _add_common_flags(p_all, hide_path=True)
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
