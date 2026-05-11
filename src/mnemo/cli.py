"""Mnemo CLI — agentic ingestion entry point.

Two subcommands feed the two knowledge axes. Each one is designed to be
scheduled (Task Scheduler / cron / GitHub Actions cron) so the corpus
stays in sync with the canonical source.

The "agentic" nature lives in the loader adapters: they decide what to
pull, normalize formats, enrich metadata, and skip uninteresting items.
For the lab we use file-system fixtures; production replaces only the
adapter (see specs_loader / bugs_loader docstrings).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mnemo.config import load_settings
from mnemo.core.models import Document
from mnemo.core.protocols import RagPipeline
from mnemo.factory import build_system
from mnemo.ingestion.bugs_loader import load_bugs
from mnemo.ingestion.specs_loader import load_specs

logger = logging.getLogger("mnemo")


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
    docs = load_specs(source)
    return _ingest(system.specs, docs, "specs")


def _cmd_bugs(args: argparse.Namespace) -> int:
    settings = load_settings()
    source: Path = args.path or settings.bugs_source_dir
    system = build_system(settings)
    docs = load_bugs(source)
    return _ingest(system.bugs, docs, "bugs")


def _cmd_all(args: argparse.Namespace) -> int:
    rc = _cmd_specs(args)
    rc |= _cmd_bugs(args)
    return rc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemo-ingest", description="Mnemo ingestion CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_specs = sub.add_parser("specs", help="Ingest specs from a folder mirror.")
    p_specs.add_argument("--path", type=Path, default=None,
                         help="Override MNEMO_SPECS_SOURCE_DIR for this run.")
    p_specs.set_defaults(func=_cmd_specs)

    p_bugs = sub.add_parser("bugs", help="Ingest resolved bugs from a folder mirror.")
    p_bugs.add_argument("--path", type=Path, default=None,
                        help="Override MNEMO_BUGS_SOURCE_DIR for this run.")
    p_bugs.set_defaults(func=_cmd_bugs)

    p_all = sub.add_parser("all", help="Ingest both axes (specs + bugs).")
    p_all.add_argument("--path", type=Path, default=None, help=argparse.SUPPRESS)
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
