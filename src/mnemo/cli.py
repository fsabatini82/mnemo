"""Mnemo CLI — agentic ingestion entry point.

Two subcommands feed the two knowledge axes. Each one is designed to be
scheduled (Task Scheduler / cron / GitHub Actions cron) so the corpus
stays in sync with the canonical source.

Two ingestion modes, opt-in per invocation:

- **deterministic** (default): the loader adapters parse files
  mechanically. Fast, predictable, no external dependencies.
- **agentic copilot** (`--agentic copilot`): an LLM-powered agent
  driven by the GitHub Copilot CLI extracts structured metadata,
  classifies items as indexable vs noise, enriches cross-references.

The default mode keeps the lab fully offline. The agentic mode shines
when you need quality classification or normalization of heterogeneous
sources — at the cost of LLM calls per item.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from mnemo.config import load_settings
from mnemo.core.models import Document
from mnemo.core.protocols import RagPipeline
from mnemo.factory import build_system
from mnemo.ingestion.bugs_loader import load_bugs
from mnemo.ingestion.specs_loader import load_specs

logger = logging.getLogger("mnemo")

AgentRuntime = Literal["copilot"]


# ---------------------------------------------------------------------------
# Mode-specific loaders
# ---------------------------------------------------------------------------


def _load_specs(source: Path, agent: AgentRuntime | None) -> list[Document]:
    if agent is None:
        return load_specs(source)
    if agent == "copilot":
        from mnemo.ingestion.agents.copilot.runner import CopilotRunner
        from mnemo.ingestion.agents.copilot.specs_agent import CopilotSpecsAgent

        runner = CopilotRunner()
        if not runner.is_available():
            logger.error(
                "Copilot CLI not found (%s). Configure MNEMO_COPILOT_BIN, "
                "or run without --agentic to fall back to the deterministic loader.",
                runner.describe(),
            )
            raise SystemExit(2)
        logger.info("Using agentic ingestion: %s", runner.describe())
        return CopilotSpecsAgent(runner=runner).ingest(source)
    raise ValueError(f"Unknown agent runtime: {agent!r}")


def _load_bugs(source: Path, agent: AgentRuntime | None) -> list[Document]:
    if agent is None:
        return load_bugs(source)
    if agent == "copilot":
        from mnemo.ingestion.agents.copilot.bugs_agent import CopilotBugsAgent
        from mnemo.ingestion.agents.copilot.runner import CopilotRunner

        runner = CopilotRunner()
        if not runner.is_available():
            logger.error(
                "Copilot CLI not found (%s). Configure MNEMO_COPILOT_BIN, "
                "or run without --agentic to fall back to the deterministic loader.",
                runner.describe(),
            )
            raise SystemExit(2)
        logger.info("Using agentic ingestion: %s", runner.describe())
        return CopilotBugsAgent(runner=runner).ingest(source)
    raise ValueError(f"Unknown agent runtime: {agent!r}")


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


def _cmd_specs(args: argparse.Namespace) -> int:
    settings = load_settings()
    source: Path = args.path or settings.specs_source_dir
    system = build_system(settings)
    docs = _load_specs(source, getattr(args, "agentic", None))
    return _ingest(system.specs, docs, "specs")


def _cmd_bugs(args: argparse.Namespace) -> int:
    settings = load_settings()
    source: Path = args.path or settings.bugs_source_dir
    system = build_system(settings)
    docs = _load_bugs(source, getattr(args, "agentic", None))
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
        "--agentic",
        choices=["copilot"],
        default=None,
        help=(
            "Use an LLM-powered ingestion agent instead of the deterministic "
            "loader. Requires the chosen runtime to be installed (e.g. the "
            "GitHub Copilot CLI for 'copilot'). Default: deterministic."
        ),
    )


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
