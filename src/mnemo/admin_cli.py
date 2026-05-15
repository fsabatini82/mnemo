"""Mnemo admin CLI — manage the project registry, scaffold specs, inspect models.

Subcommands:
  list-projects           Show registered projects, their IDs and envs.
  rename-project          Rename a project slug (preserves underlying ID).
  drop-project            Remove a project (registry + collections).
  show-collection-names   Resolve and print the effective collection names.
  new-spec                Scaffold a new spec from a canonical template
                          (story | adr | epic).
  list-models             List configured family aliases + curated short-names
                          + cross-check against the live GH Models catalog.
  test-runtime            Send a sanity prompt through the selected runtime
                          and report latency + token usage.

The registry lives at `<MNEMO_PERSIST_DIR>/projects.json`. Underlying
collection drops happen in Chroma/LanceDB and are best-effort.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from mnemo.config import load_settings
from mnemo.ingestion.agents.runner_factory import RunnerBuildError, build_runner
from mnemo.models_catalog import (
    FAMILY_TO_SETTING,
    MODEL_SHORTNAMES,
    fetch_catalog_models,
    list_shortnames,
    resolve_model,
    resolve_token_from_env,
)
from mnemo.registry import (
    ENVIRONMENTS,
    ProjectRegistry,
    RegistryError,
    collection_name,
    open_registry,
    validate_environment,
    validate_slug,
)
from mnemo.templates_io import TEMPLATE_KINDS, TemplateError, render as render_template

logger = logging.getLogger("mnemo-admin")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    settings = load_settings()
    registry = open_registry(settings.persist_dir)
    records = registry.projects()
    if not records:
        print(f"No projects registered at {registry._path}.")  # noqa: SLF001
        print("Run `mnemo-ingest specs --project <slug>` to register one.")
        return 0
    print(f"{'ID':<5} {'SLUG':<32} {'ENVIRONMENTS':<24} CREATED")
    print("-" * 80)
    for r in records:
        envs = ",".join(r.environments) or "—"
        print(f"{r.id:<5} {r.slug:<32} {envs:<24} {r.created}")
    return 0


def _cmd_rename(args: argparse.Namespace) -> int:
    settings = load_settings()
    registry = open_registry(settings.persist_dir)
    try:
        record = registry.rename(args.old_slug, args.new_slug)
        registry.save()
    except RegistryError as exc:
        logger.error(str(exc))
        return 2
    print(f"Renamed {args.old_slug!r} → {args.new_slug!r} (id={record.id}, "
          f"storage prefix unchanged).")
    return 0


def _cmd_drop(args: argparse.Namespace) -> int:
    settings = load_settings()
    registry = open_registry(settings.persist_dir)
    record = registry.get(args.slug)
    if record is None:
        logger.error("Project %r not registered.", args.slug)
        return 2

    envs_to_drop = [args.environment] if args.environment else list(record.environments)
    if not envs_to_drop:
        # Slug registered but no environments seen yet — just drop the entry.
        registry.drop(args.slug)
        registry.save()
        print(f"Dropped project {args.slug!r} from registry (no collections existed).")
        return 0

    # Drop underlying store collections.
    dropped_collections: list[str] = []
    failures: list[tuple[str, str]] = []
    for env in envs_to_drop:
        for axis in (settings.specs_collection, settings.bugs_collection):
            name = collection_name(record.id, env, axis)
            try:
                _drop_collection(settings, name)
                dropped_collections.append(name)
            except Exception as exc:  # noqa: BLE001
                failures.append((name, str(exc)))

    if args.environment:
        # Single-env drop: keep the project record, just remove the env.
        record.environments = [e for e in record.environments if e != args.environment]
        registry._dirty = True  # noqa: SLF001
        if not record.environments:
            registry.drop(args.slug)
        registry.save()
    else:
        # Whole-project drop: remove the entry entirely.
        registry.drop(args.slug)
        registry.save()

    print(f"Dropped {len(dropped_collections)} collection(s):")
    for name in dropped_collections:
        print(f"  - {name}")
    if failures:
        print(f"WARNINGS — {len(failures)} drop(s) failed:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
    return 0


def _cmd_new_spec(args: argparse.Namespace) -> int:
    """Render a template to file or stdout."""
    try:
        rendered = render_template(args.kind, id=args.id, title=args.title)
    except TemplateError as exc:
        logger.error(str(exc))
        return 2

    if args.output is None:
        # Print to stdout for `>` redirection convenience.
        print(rendered)
        return 0

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        logger.error(
            "%s already exists. Pass --force to overwrite.", out_path
        )
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.kind} template → {out_path}")
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    """Show configured family aliases + short-names, cross-check live catalog."""
    settings = load_settings()

    # Pre-fetch the live catalog (if token available). Catalog may be empty
    # on auth/network issues — that's fine, we degrade gracefully.
    token = resolve_token_from_env()
    catalog: set[str] = set()
    if token:
        live = fetch_catalog_models(token, settings.ghmodels_catalog_endpoint)
        catalog = set(live)
    else:
        print("(no token — skipping live catalog cross-check)")
        print("set MNEMO_GHMODELS_TOKEN / GH_TOKEN / GITHUB_TOKEN to enable it\n")

    print("Configured family aliases:")
    for alias, setting_attr in FAMILY_TO_SETTING.items():
        raw = getattr(settings, setting_attr, "")
        try:
            full = resolve_model(alias, settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  {alias:<8} → (unresolved: {exc})")
            continue
        mark = _catalog_mark(full, catalog)
        print(f"  {alias:<8} → {full:<40} (setting: {raw}) {mark}")

    print("\nMnemo curated short-names:")
    for short in list_shortnames():
        full = MODEL_SHORTNAMES[short]
        mark = _catalog_mark(full, catalog)
        print(f"  {short:<20} → {full:<40} {mark}")

    if catalog:
        known = set(MODEL_SHORTNAMES.values())
        extras = sorted(catalog - known)
        if extras:
            print("\nAvailable in catalog but not in short-names "
                  "(use --agentic <full-id> to access):")
            for ident in extras:
                print(f"  {ident}")
    return 0


def _catalog_mark(full_id: str, catalog: set[str]) -> str:
    if not catalog:
        return ""
    return "[✓ in catalog]" if full_id in catalog else "[✗ not in catalog]"


def _cmd_test_runtime(args: argparse.Namespace) -> int:
    """Send a sanity prompt through the selected runtime."""
    import time

    settings = load_settings()
    agentic_value = args.agentic or settings.ghmodels_model
    effort = args.reasoning_effort or settings.ghmodels_reasoning_effort
    try:
        runner = build_runner(agentic_value, settings, reasoning_effort=effort)
    except RunnerBuildError as exc:
        logger.error("%s", exc)
        return 2

    if not runner.is_available():
        logger.error("Runner not ready: %s", runner.describe())
        return 2

    print(f"Runner: {runner.describe()}")
    prompt = "Reply with the single word 'ok' and nothing else."
    started = time.monotonic()
    try:
        content = runner.run(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Runtime call failed: %s", exc)
        return 1
    elapsed = time.monotonic() - started

    preview = content.strip()[:200]
    print(f"Latency:        {elapsed:.2f}s")
    print(f"Content bytes:  {len(content)}")
    print(f"Content preview: {preview!r}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Resolve the effective collection names for a project/env combo."""
    settings = load_settings()
    registry = open_registry(settings.persist_dir)
    record = registry.get(args.slug)
    if record is None:
        logger.error("Project %r not registered. Run `mnemo-ingest` first.", args.slug)
        return 2
    env = args.environment or settings.environment
    print(f"project:        {record.slug} (id={record.id})")
    print(f"environment:    {env}")
    print(f"specs:          {collection_name(record.id, env, settings.specs_collection)}")
    print(f"bug_memory:     {collection_name(record.id, env, settings.bugs_collection)}")
    return 0


def _drop_collection(settings: Any, name: str) -> None:
    """Best-effort drop of a collection from the active store backend."""
    if settings.store == "chroma":
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(settings.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        try:
            client.delete_collection(name=name)
        except Exception:
            # Already gone or never existed — treat as success.
            pass
        return

    if settings.store == "lance":
        try:
            import lancedb
        except ImportError:
            return
        db = lancedb.connect(str(settings.persist_dir))
        # LanceStore uses `{collection}_chunks` table naming convention.
        table_name = f"{name}_chunks"
        if table_name in db.table_names():
            db.drop_table(table_name)
        return

    # Unknown backend — silent no-op.


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    try:
        return validate_slug(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnemo-admin",
        description="Mnemo admin CLI — manage the project registry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-projects", help="List all registered projects.")
    p_list.set_defaults(func=_cmd_list)

    p_rename = sub.add_parser("rename-project", help="Rename a project slug.")
    p_rename.add_argument("old_slug", type=_slug)
    p_rename.add_argument("new_slug", type=_slug)
    p_rename.set_defaults(func=_cmd_rename)

    p_drop = sub.add_parser(
        "drop-project",
        help="Drop a project (registry entry + underlying collections).",
    )
    p_drop.add_argument("slug", type=_slug)
    p_drop.add_argument(
        "--env",
        dest="environment",
        choices=list(ENVIRONMENTS),
        default=None,
        help="Drop only this environment. Default: drop all.",
    )
    p_drop.set_defaults(func=_cmd_drop)

    p_show = sub.add_parser(
        "show-collection-names",
        help="Resolve the effective collection names for a project/env.",
    )
    p_show.add_argument("slug", type=_slug)
    p_show.add_argument("--env", dest="environment",
                        choices=list(ENVIRONMENTS), default=None)
    p_show.set_defaults(func=_cmd_show)

    p_new = sub.add_parser(
        "new-spec",
        help="Scaffold a new spec from a canonical template (story|adr|epic).",
    )
    p_new.add_argument(
        "kind", choices=list(TEMPLATE_KINDS),
        help="Template kind to instantiate.",
    )
    p_new.add_argument("--id", required=True,
                       help="Spec identifier, e.g. US-205 or ADR-007.")
    p_new.add_argument("--title", required=True,
                       help="One-line title to place in the heading.")
    p_new.add_argument("-o", "--output", default=None,
                       help="Write to this file path. If omitted, print to stdout.")
    p_new.add_argument("--force", action="store_true",
                       help="Overwrite the output file if it already exists.")
    p_new.set_defaults(func=_cmd_new_spec)

    p_list_models = sub.add_parser(
        "list-models",
        help="List configured family aliases + short-names + live catalog cross-check.",
    )
    p_list_models.set_defaults(func=_cmd_list_models)

    p_test = sub.add_parser(
        "test-runtime",
        help="Send a sanity prompt through the selected runtime; report latency + tokens.",
    )
    p_test.add_argument(
        "--agentic",
        nargs="?",
        const=None,
        default=None,
        metavar="MODEL",
        help=(
            "Runtime to test. Without value: uses MNEMO_GHMODELS_MODEL. "
            "`copilot`: subprocess CLI. Or any short-name / family alias "
            "/ full id."
        ),
    )
    p_test.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Override reasoning_effort for this test call.",
    )
    p_test.set_defaults(func=_cmd_test_runtime)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
