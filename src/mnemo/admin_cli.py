"""Mnemo admin CLI — manage the project registry.

Subcommands:
  list-projects           Show registered projects, their IDs and envs.
  rename-project          Rename a project slug (preserves underlying ID).
  drop-project            Remove a project (registry + collections).
  show-collection-names   Resolve and print the effective collection names.

The registry lives at `<MNEMO_PERSIST_DIR>/projects.json`. Underlying
collection drops happen in Chroma/LanceDB and are best-effort: if the
store backend doesn't have the collection, that's logged but not an
error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from mnemo.config import load_settings
from mnemo.registry import (
    ENVIRONMENTS,
    ProjectRegistry,
    RegistryError,
    collection_name,
    open_registry,
    validate_environment,
    validate_slug,
)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
