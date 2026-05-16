"""Mnemo MCP server — exposes organizational memory to IDE agents.

Three knowledge axes:
  • specs       — what to build (user stories, ADRs, epics) — human-authored
  • bug_memory  — what went wrong before (resolved bugs with root causes)
  • devops      — work-in-flight (Features, PBI, open bugs) mirrored from
                  Azure DevOps; state is authoritative verbatim from DevOps

Project + environment isolation: a single `mnemo-server` instance is
scoped to one `(project, environment)` pair via CLI flags or env vars,
so an IDE workspace can register `mnemo-server --project alpha --env prd`
and an entirely separate workspace can register the same binary with
different flags pointing at different storage.

Tools are intentionally narrow and composable so an agent can chain
them during multi-step reasoning.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mnemo.audit import AuditEngine, _worst_severity
from mnemo.audit_deep import DeepAuditEngine
from mnemo.config import Settings, load_settings
from mnemo.env_file import commit_change
from mnemo.factory import MnemoSystem, build_system
from mnemo.ingestion.agents.runner_factory import RunnerBuildError, build_runner
from mnemo.lifecycle import assert_canonical as _assert_canonical_lifecycle
from mnemo.models_catalog import resolve_token_from_env
from mnemo.registry import ENVIRONMENTS
from mnemo.runtime_config import (
    CATEGORY,
    ConfigCategory,
    InvalidConfigValue,
    UnknownConfigKey,
    classify,
    validate_value,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("mnemo")

# Lazy-initialized at main() to allow CLI flags to override env before
# Settings are constructed. Tools assert these are populated.
_settings: Settings | None = None
_system: MnemoSystem | None = None


# ---------------------------------------------------------------------------
# Specs side — "what to build"
# ---------------------------------------------------------------------------


@mcp.tool()
def query_specs(
    question: str,
    k: int | None = None,
    lifecycle: str | None = None,
) -> dict[str, Any]:
    """Semantic search across the project specs (user stories, ADRs, epics).

    Use this when implementing a feature: it surfaces the acceptance
    criteria, related ADRs, and any cross-references the spec carries.

    Args:
        question: Natural-language question or spec identifier to look up.
        k: Optional override for the number of chunks to return.
        lifecycle: Optional filter — restrict results to specs whose
            `lifecycle` metadata equals this value. Must be one of
            `proposed`, `in-progress`, `implemented`, `superseded`, `as-is`.
    """
    settings, system = _require_loaded()
    where = None
    if lifecycle is not None:
        _assert_canonical_lifecycle(lifecycle)
        where = {"lifecycle": lifecycle}
    result = system.specs.query(question, k=k or settings.top_k, where=where)
    return {
        "axis": "specs",
        "project": settings.project,
        "environment": settings.environment,
        "lifecycle_filter": lifecycle,
        "question": result.question,
        "hits": [_hit_to_dict(h) for h in result.hits],
    }


@mcp.tool()
def get_spec(spec_id: str) -> dict[str, Any]:
    """Retrieve a spec by its identifier (e.g. "US-102", "ADR-002")."""
    return query_specs(spec_id, k=10)


# ---------------------------------------------------------------------------
# Bug memory side — "what went wrong before"
# ---------------------------------------------------------------------------


@mcp.tool()
def query_bugs(symptom: str, k: int | None = None) -> dict[str, Any]:
    """Semantic search across the resolved bug history.

    Args:
        symptom: Description of the issue, error message, or area of concern.
        k: Optional override for the number of bugs to return.
    """
    settings, system = _require_loaded()
    result = system.bugs.query(symptom, k=k or settings.top_k)
    return {
        "axis": "bug_memory",
        "project": settings.project,
        "environment": settings.environment,
        "question": result.question,
        "hits": [_hit_to_dict(h) for h in result.hits],
    }


@mcp.tool()
def get_bug(bug_id: str) -> dict[str, Any]:
    """Retrieve a bug by its identifier (e.g. "BUG-503")."""
    return query_bugs(bug_id, k=3)


# ---------------------------------------------------------------------------
# DevOps side — "what is in-flight"
# ---------------------------------------------------------------------------


@mcp.tool()
def query_devops(
    question: str,
    k: int | None = None,
    state: str | None = None,
    work_item_type: str | None = None,
) -> dict[str, Any]:
    """Semantic search across DevOps work items mirrored from Azure DevOps.

    Covers Features, Product Backlog Items, and *open* Bugs. Resolved
    bugs live in the `bug_memory` axis. State values are verbatim from
    DevOps (including custom process-template values like "Validato BU"
    or "Progettato") — pass them as-is.

    Args:
        question: Natural-language question or work-item identifier
            (e.g. "WI-396679").
        k: Optional override for the number of chunks to return.
        state: Optional filter — restrict to work items whose `state`
            metadata equals this value (verbatim DevOps state).
        work_item_type: Optional filter — restrict to a specific type
            (e.g. "feature", "product-backlog-item", "bug").
    """
    settings, system = _require_loaded()
    where: dict[str, Any] = {}
    if state is not None:
        where["state"] = state
    if work_item_type is not None:
        where["work_item_type"] = work_item_type
    result = system.devops.query(
        question, k=k or settings.top_k, where=where or None,
    )
    return {
        "axis": "devops",
        "project": settings.project,
        "environment": settings.environment,
        "state_filter": state,
        "work_item_type_filter": work_item_type,
        "question": result.question,
        "hits": [_hit_to_dict(h) for h in result.hits],
    }


@mcp.tool()
def get_devops_item(work_item_id: str) -> dict[str, Any]:
    """Retrieve a DevOps work item by its identifier (e.g. "WI-396679")."""
    return query_devops(work_item_id, k=3)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@mcp.tool()
def audit_spec(
    spec_id: str,
    deep: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Run drift checks against a single spec.

    By default runs the cheap deterministic checks (status / coverage /
    template). Pass `deep=True` to also run the LLM-powered behavior
    drift check — slower but catches divergences cheap checks can't see.

    Args:
        spec_id: Spec identifier (e.g. "US-102").
        deep: When True, also run the behavior audit via an LLM.
        model: Optional model override for the deep audit. Accepts
            short-names ("gpt-5", "gpt-5-mini", "claude-opus-4-6"),
            family aliases ("gpt", "claude", "opus", "sonnet"), full
            ids ("openai/gpt-5"), or "copilot" for the subprocess CLI.
            Falls back to MNEMO_GHMODELS_MODEL when omitted.
    """
    settings, system = _require_loaded()
    engine = AuditEngine(system.specs, code_root=settings.code_root)
    report = engine.audit_spec(spec_id)
    if report is None:
        return {
            "spec_id": spec_id,
            "error": f"Spec {spec_id!r} not found in the active project's specs collection.",
        }
    if deep:
        deep_engine = _build_deep_engine(settings, system, model)
        if deep_engine is None:
            report.issues.append(_runner_unavailable_issue(model))
        else:
            deep_issues = deep_engine.audit_spec(spec_id)
            if deep_issues:
                report.issues.extend(deep_issues)
                report.severity = _worst_severity(report.issues)
    return report.to_dict()


@mcp.tool()
def audit_spec_behavior(spec_id: str, model: str | None = None) -> dict[str, Any]:
    """Run only the LLM-powered behavior drift check (deep audit).

    Use this when the cheap checks already passed but you suspect the
    code's actual behavior may have drifted from the spec semantics.

    Args:
        spec_id: Spec identifier (e.g. "US-102").
        model: Optional model override. Same options as `audit_spec`.
            Default: MNEMO_GHMODELS_MODEL.
    """
    settings, system = _require_loaded()
    deep_engine = _build_deep_engine(settings, system, model)
    if deep_engine is None:
        return {
            "spec_id": spec_id,
            "error": (
                f"Runner unavailable for model={model or settings.ghmodels_model!r}. "
                "Set MNEMO_GHMODELS_TOKEN (or GH_TOKEN/GITHUB_TOKEN) for "
                "GitHub Models, or pass model='copilot' for the subprocess CLI."
            ),
        }
    issues = deep_engine.audit_spec(spec_id)
    return {
        "spec_id": spec_id,
        "issues": [i.to_dict() for i in issues],
        "has_drift": bool(issues),
        "severity": _worst_severity(issues),
    }


def _build_deep_engine(
    settings: Settings,
    system: MnemoSystem,
    model: str | None,
) -> DeepAuditEngine | None:
    """Build the deep-audit engine with optional per-call model override."""
    agentic_value = model or settings.ghmodels_model
    try:
        runner = build_runner(agentic_value, settings)
    except RunnerBuildError as exc:
        logger.warning("Deep audit: %s", exc)
        return None
    engine = DeepAuditEngine(system.specs, code_root=settings.code_root, runner=runner)
    if not engine.is_available():
        logger.warning("Deep audit runner not available: %s", runner.describe())
        return None
    return engine


def _runner_unavailable_issue(model: str | None) -> Any:
    from mnemo.audit import DriftIssue
    return DriftIssue(
        type="behavior",
        severity="low",
        description=(
            "Deep audit requested but the LLM runner is not available "
            f"(requested model: {model or 'default'}). Cheap-only results returned."
        ),
        suggested_action=(
            "Set MNEMO_GHMODELS_TOKEN (or GH_TOKEN/GITHUB_TOKEN) for "
            "GitHub Models, or configure MNEMO_COPILOT_BIN if you intended "
            "to use the subprocess Copilot CLI."
        ),
    )


@mcp.tool()
def audit_implemented_specs() -> dict[str, Any]:
    """Run cheap drift checks across every `lifecycle=implemented` spec.

    Returns a summary plus per-spec reports. Useful for periodic
    "what's drifted lately?" sweeps during refactoring or code review.
    """
    settings, system = _require_loaded()
    engine = AuditEngine(system.specs, code_root=settings.code_root)
    reports = engine.audit_all(lifecycle="implemented")
    return {
        "project": settings.project,
        "environment": settings.environment,
        "total_audited": len(reports),
        "with_drift": sum(1 for r in reports if r.has_drift),
        "reports": [r.to_dict() for r in reports],
    }


@mcp.tool()
def list_runtime_config() -> dict[str, Any]:
    """Show current Mnemo settings, categorized by mutability.

    Returns four buckets:
      - editable_now           : safe to change from chat; takes effect
                                 on the next MCP tool call (no restart).
      - editable_with_restart  : structural — change requires the server
                                 to be restarted to take effect. Mnemo
                                 also requires confirm_structural=True
                                 in set_runtime_config to write these.
      - readonly               : scope or security-sensitive. Refused
                                 by set_runtime_config.
      - hidden                 : token-shaped. Names are listed only if
                                 the env var is currently set; values
                                 are NEVER returned.
    """
    settings, _ = _require_loaded()
    buckets: dict[str, Any] = {
        "editable_now": {},
        "editable_with_restart": {},
        "readonly": {},
        "hidden": [],
    }
    for key, cat in CATEGORY.items():
        if cat == ConfigCategory.HIDDEN:
            if os.environ.get(key):
                buckets["hidden"].append(key)
            continue
        current = _current_value_for(key, settings)
        buckets[cat.value][key] = current
    return buckets


@mcp.tool()
def set_runtime_config(
    key: str,
    value: str,
    confirm_structural: bool = False,
) -> dict[str, Any]:
    """Update a Mnemo configuration value in both the live process and `.env`.

    Behavior depends on the key's category (see `list_runtime_config`):

    - **editable_now**: applied immediately. `os.environ[key]` is
      updated, the `.env` file is rewritten with an audit comment, and
      the cached Settings are reloaded so subsequent tool calls see the
      new value.
    - **editable_with_restart**: requires `confirm_structural=True`.
      Writes to `.env` and `os.environ`, but the running system keeps
      using the old value until `mnemo-server` is restarted.
    - **readonly / hidden / unknown**: refused with a clear error.

    Args:
        key: Full env var name (e.g. `MNEMO_GHMODELS_MODEL`).
        value: New value as a string.
        confirm_structural: Required = True for `editable_with_restart`
            keys, ignored for others.
    """
    settings, _ = _require_loaded()

    try:
        cat = classify(key)
    except UnknownConfigKey as exc:
        return {"error": str(exc), "key": key}

    if cat == ConfigCategory.HIDDEN:
        return {
            "error": (
                f"Key {key!r} is a token. Tokens must be exported in the "
                "shell environment, not set via MCP."
            ),
            "key": key, "category": cat.value,
        }
    if cat == ConfigCategory.READONLY:
        return {
            "error": (
                f"Key {key!r} is read-only via MCP "
                "(scope/security-sensitive)."
            ),
            "key": key, "category": cat.value,
        }
    if cat == ConfigCategory.EDITABLE_WITH_RESTART and not confirm_structural:
        return {
            "error": (
                f"Key {key!r} is structural and requires "
                "`confirm_structural=True` (the change will only take "
                "effect after restarting mnemo-server)."
            ),
            "key": key, "category": cat.value,
        }

    try:
        validate_value(key, value)
    except (InvalidConfigValue, UnknownConfigKey) as exc:
        return {"error": str(exc), "key": key, "category": cat.value}

    env_path = (settings.persist_dir.parent / ".env").resolve()
    # Prefer the cwd-relative `.env` if it exists (typical for installs);
    # fall back to persist_dir/.env (uncommon, but explicit).
    cwd_env = (Path.cwd() / ".env").resolve()
    target = cwd_env if cwd_env.is_file() else env_path

    try:
        old_value = commit_change(target, key, value)
    except OSError as exc:
        return {
            "error": f"Failed to write {target}: {exc}",
            "key": key, "category": cat.value,
        }

    # Apply to the live process env so subsequent runner calls pick it up.
    os.environ[key] = value

    takes_effect = "immediately" if cat == ConfigCategory.EDITABLE_NOW else "on_restart"
    if cat == ConfigCategory.EDITABLE_NOW:
        # Reload cached settings so Mnemo's own tools see the new value.
        global _settings
        _settings = load_settings()

    return {
        "key": key,
        "old_value": old_value,
        "new_value": value,
        "category": cat.value,
        "takes_effect": takes_effect,
        "env_file": str(target),
    }


@mcp.tool()
def mnemo_info() -> dict[str, Any]:
    """Return the active configuration — useful for sanity checks in demos."""
    settings, system = _require_loaded()
    return {
        "project": settings.project,
        "project_id": system.project_id,
        "environment": settings.environment,
        "effective_prefix": system.effective_prefix,
        "store": settings.store,
        "pipeline": settings.pipeline,
        "embed_model": settings.embed_model,
        "collections": {
            "specs": f"{system.effective_prefix}_{settings.specs_collection}",
            "bugs": f"{system.effective_prefix}_{settings.bugs_collection}",
            "devops": f"{system.effective_prefix}_{settings.devops_collection}",
        },
        "sources": {
            "specs_dir": str(settings.specs_source_dir),
            "bugs_dir": str(settings.bugs_source_dir),
            "devops_dir": str(settings.devops_source_dir),
        },
        "chunking": {
            "size": settings.chunk_size,
            "overlap": settings.chunk_overlap,
        },
        "top_k": settings.top_k,
        "ghmodels": {
            "model": settings.ghmodels_model,
            "reasoning_effort": settings.ghmodels_reasoning_effort,
            "max_completion_tokens": settings.ghmodels_max_completion_tokens,
            "token_present": resolve_token_from_env() is not None,
        },
        "family_aliases": {
            "gpt": settings.family_gpt,
            "claude": settings.family_claude,
            "sonnet": settings.family_sonnet,
            "opus": settings.family_opus,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_value_for(key: str, settings: Settings) -> str:
    """Return the live value for `key`, preferring os.environ, then Settings.

    Settings field names are the env var stripped of `MNEMO_` prefix and
    lowercased (pydantic-settings convention).
    """
    if value := os.environ.get(key):
        return value
    if key.startswith("MNEMO_"):
        field_name = key[len("MNEMO_"):].lower()
        if field_name in Settings.model_fields:
            return str(getattr(settings, field_name))
    return ""


def _hit_to_dict(h: Any) -> dict[str, Any]:
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "score": h.score,
        "text": h.text,
        "metadata": h.metadata,
    }


def _require_loaded() -> tuple[Settings, MnemoSystem]:
    if _settings is None or _system is None:
        raise RuntimeError(
            "mnemo-server tools were called before main() initialized the system."
        )
    return _settings, _system


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_TTY_USAGE_HINT = """\
mnemo-server is an MCP stdio server.

It is meant to be launched by an MCP client (e.g. GitHub Copilot via
.vscode/mcp.json, or Claude Code via `claude mcp add`), not from an
interactive terminal — stdin must carry JSON-RPC messages.

To verify the install instead, run:
  mnemo-ingest --help        (real CLI with --help support)
  mnemo-admin list-projects  (registry inspection)
  where mnemo-server         (Windows: print the executable path)
  which mnemo-server         (Linux/macOS: print the executable path)

Override with --force-stdio if you really want to start the stdio loop
manually (e.g. to debug a client).
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemo-server",
        description="MCP stdio server exposing the Mnemo organizational memory.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override MNEMO_PROJECT for this server instance.",
    )
    parser.add_argument(
        "--env",
        dest="environment",
        choices=list(ENVIRONMENTS),
        default=None,
        help="Override MNEMO_ENVIRONMENT for this server instance.",
    )
    parser.add_argument(
        "--force-stdio",
        action="store_true",
        help="Bypass the TTY-guard and run the stdio loop unconditionally.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()

    if args.project:
        os.environ["MNEMO_PROJECT"] = args.project
    if args.environment:
        os.environ["MNEMO_ENVIRONMENT"] = args.environment

    # Friendly guard against interactive launch (stdin = TTY).
    if sys.stdin.isatty() and not args.force_stdio:
        sys.stderr.write(_TTY_USAGE_HINT)
        sys.exit(0)

    global _settings, _system
    _settings = load_settings()
    _system = build_system(_settings)  # read-only: auto_register=False
    logger.info(
        "Mnemo server scope: project=%s (id=%s) env=%s",
        _settings.project, _system.project_id, _settings.environment,
    )

    mcp.run()


if __name__ == "__main__":
    main()
