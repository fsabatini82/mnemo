"""Mnemo audit CLI — drift detection commands.

Subcommands:
  drift   Run the cheap drift checks against the active project's specs.

The cheap path looks at three signals (see `mnemo.audit` docstring):
- status drift (implemented spec, files missing on disk)
- coverage drift (spec ID mentioned by files not declared, or declared
  files that don't mention the ID)
- template drift (spec not aligned with the canonical template)

Output is JSON when `--output` is supplied, or a human-readable
summary on stdout otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mnemo.audit import AuditEngine, DriftReport
from mnemo.config import load_settings
from mnemo.factory import build_system
from mnemo.lifecycle import LIFECYCLE_VALUES, assert_canonical
from mnemo.registry import ENVIRONMENTS

logger = logging.getLogger("mnemo-audit")


def _apply_scope_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "project", None):
        os.environ["MNEMO_PROJECT"] = args.project
    if getattr(args, "environment", None):
        os.environ["MNEMO_ENVIRONMENT"] = args.environment
    if getattr(args, "code_root", None):
        os.environ["MNEMO_CODE_ROOT"] = str(args.code_root)


def _cmd_drift(args: argparse.Namespace) -> int:
    _apply_scope_overrides(args)
    settings = load_settings()
    system = build_system(settings)
    engine = AuditEngine(system.specs, code_root=settings.code_root)

    if args.spec:
        report = engine.audit_spec(args.spec)
        if report is None:
            logger.error("Spec %r not found in project=%s env=%s.",
                         args.spec, settings.project, settings.environment)
            return 2
        reports = [report]
    else:
        lifecycle = args.lifecycle
        if lifecycle:
            assert_canonical(lifecycle)
        reports = engine.audit_all(lifecycle=lifecycle)

    payload: dict[str, Any] = {
        "project": settings.project,
        "project_id": system.project_id,
        "environment": settings.environment,
        "code_root": str(Path(settings.code_root).resolve()),
        "lifecycle_filter": args.lifecycle,
        "spec_filter": args.spec,
        "total_specs_audited": len(reports),
        "specs_with_drift": sum(1 for r in reports if r.has_drift),
        "reports": [r.to_dict() for r in reports],
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        print(f"Wrote drift report → {args.output}")
    else:
        _print_human(payload, reports)

    # Exit non-zero if any high-severity drift, so this can be wired into CI.
    if any(r.severity == "high" for r in reports):
        return 1
    return 0


def _print_human(payload: dict[str, Any], reports: list[DriftReport]) -> None:
    print(
        f"Audit: project={payload['project']} (id={payload['project_id']}) "
        f"env={payload['environment']}"
    )
    print(f"Code root: {payload['code_root']}")
    print(
        f"Specs audited: {payload['total_specs_audited']} · "
        f"with drift: {payload['specs_with_drift']}"
    )
    print("-" * 70)
    if not reports:
        print("(no specs matched)")
        return
    for r in sorted(reports, key=lambda x: (_severity_key(x.severity), x.spec_id)):
        marker = {"none": "  ", "low": "· ", "medium": "› ", "high": "‼ "}.get(r.severity, "  ")
        print(f"{marker}{r.spec_id:<14} [{r.severity:<6}] {r.kind:<8} "
              f"lifecycle={r.lifecycle:<14} template={r.template_compliance}")
        for issue in r.issues:
            print(f"    [{issue.type}] {issue.description}")


def _severity_key(s: str) -> int:
    return {"high": 0, "medium": 1, "low": 2, "none": 3}.get(s, 4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemo-audit",
        description="Mnemo audit CLI — drift detection and related checks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_drift = sub.add_parser("drift", help="Run cheap drift checks against the active project's specs.")
    p_drift.add_argument(
        "--project", default=None,
        help="Override MNEMO_PROJECT for this run.",
    )
    p_drift.add_argument(
        "--env", dest="environment", choices=list(ENVIRONMENTS), default=None,
        help="Override MNEMO_ENVIRONMENT for this run.",
    )
    p_drift.add_argument(
        "--code-root", type=Path, default=None,
        help="Path to the working tree against which `related_files` are resolved. "
             "Default: from MNEMO_CODE_ROOT env / Settings (CWD if unset).",
    )
    p_drift.add_argument(
        "--spec", default=None,
        help="Audit only this spec ID (e.g. US-102). Default: all specs.",
    )
    p_drift.add_argument(
        "--lifecycle", choices=list(LIFECYCLE_VALUES), default=None,
        help="Audit only specs with this lifecycle value.",
    )
    p_drift.add_argument(
        "-o", "--output", default=None,
        help="Write JSON report to this file. Default: human summary on stdout.",
    )
    p_drift.set_defaults(func=_cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
