"""Project registry — maps human slugs to stable 3-digit IDs.

The registry persists at `<persist_dir>/projects.json` and is the
source of truth for which projects exist and which environments each
one has seen ingestion for.

Rationale for numeric IDs (vs slug-direct collection naming):

- **Rename safety**: renaming "alpha" to "alpha-platform-team" is a
  JSON edit, not a DB migration. The underlying collections keep the
  same `001_*` prefix.
- **Uniform length**: collection names stay short (`001_prd_specs`)
  regardless of how long the human-facing slug is.
- **Predictable ordering**: registries can be sorted and audited
  numerically.

Auto-registration happens on the first `mnemo-ingest` for a new slug.
`mnemo-server` is read-only with respect to the registry — if the
configured project doesn't exist yet, the server logs a warning but
still starts.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Slug rules: lowercase, must start with a letter, max 32 chars.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

# Environment enum kept here so callers don't have to import Settings.
ENVIRONMENTS = ("dev", "col", "pre", "prd")

_REGISTRY_FILENAME = "projects.json"
_REGISTRY_VERSION = 1
_DEFAULT_PROJECT_SLUG = "demo-project"


class RegistryError(RuntimeError):
    """Raised on invalid slug, invalid env, missing project, etc."""


@dataclass(slots=True)
class ProjectRecord:
    id: str
    slug: str
    created: str
    environments: list[str] = field(default_factory=list)
    description: str = ""

    def add_environment(self, env: str) -> None:
        if env not in self.environments:
            self.environments.append(env)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_slug(slug: str) -> str:
    if not _SLUG_RE.fullmatch(slug):
        raise RegistryError(
            f"Invalid project slug {slug!r}: must match {_SLUG_RE.pattern} "
            "(lowercase, start with a letter, max 32 chars, [a-z0-9-] only)."
        )
    return slug


def validate_environment(env: str) -> str:
    if env not in ENVIRONMENTS:
        raise RegistryError(
            f"Invalid environment {env!r}: must be one of {ENVIRONMENTS}."
        )
    return env


def collection_name(project_id: str, env: str, axis: str) -> str:
    """Build the canonical collection/table name shared by Chroma + LanceDB.

    `<id 3 digits>_<env>_<axis>` — underscore-only, filesystem-safe,
    fits in Chroma's 63-char and LanceDB's 64-char limits with margin.
    """
    if not re.fullmatch(r"\d{3}", project_id):
        raise RegistryError(f"Invalid project_id {project_id!r}: must be 3 digits.")
    validate_environment(env)
    if not re.fullmatch(r"[a-z_]+", axis):
        raise RegistryError(f"Invalid axis {axis!r}: must be lowercase letters/underscore.")
    return f"{project_id}_{env}_{axis}"


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------


class ProjectRegistry:
    """File-backed registry of `slug → ProjectRecord`.

    Concurrency: a process-local lock guards reads and writes. For
    cross-process safety we rely on the registry being rewritten in
    full each time (atomic via temp-file rename); concurrent writes
    from two processes are not strictly serialized, but the small
    payload size (a JSON object with N projects) makes torn writes
    practically impossible at the IO level. Mnemo is single-user by
    default — fix if you ever deploy it as a multi-tenant service.
    """

    _lock = threading.Lock()

    def __init__(self, persist_dir: Path) -> None:
        self._persist_dir = persist_dir
        self._path = persist_dir / _REGISTRY_FILENAME
        self._projects: dict[str, ProjectRecord] = {}
        self._dirty = False

    # ------------------------------------------------------------------ load

    def load(self) -> "ProjectRegistry":
        if not self._path.is_file():
            return self
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"projects.json is corrupted: {exc}") from exc

        if not isinstance(raw, dict) or raw.get("version") != _REGISTRY_VERSION:
            raise RegistryError(
                f"projects.json version mismatch: expected {_REGISTRY_VERSION}, "
                f"got {raw.get('version')!r}"
            )

        projects = raw.get("projects") or {}
        if not isinstance(projects, dict):
            raise RegistryError("projects.json: `projects` must be an object.")

        for slug, payload in projects.items():
            validate_slug(slug)
            record = ProjectRecord(
                id=str(payload["id"]),
                slug=slug,
                created=str(payload.get("created", "")),
                environments=list(payload.get("environments") or []),
                description=str(payload.get("description", "")),
            )
            # Validate referenced envs.
            for env in record.environments:
                validate_environment(env)
            self._projects[slug] = record
        return self

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        if not self._dirty:
            return
        with self._lock:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": _REGISTRY_VERSION,
                "projects": {
                    slug: {
                        "id": record.id,
                        "created": record.created,
                        "environments": list(record.environments),
                        "description": record.description,
                    }
                    for slug, record in sorted(self._projects.items())
                },
            }
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = False

    # ------------------------------------------------------------------ ops

    def projects(self) -> list[ProjectRecord]:
        return [self._projects[s] for s in sorted(self._projects)]

    def get(self, slug: str) -> ProjectRecord | None:
        validate_slug(slug)
        return self._projects.get(slug)

    def ensure(self, slug: str, *, description: str = "") -> ProjectRecord:
        """Return the record for `slug`, registering a new one if missing.

        Newly registered projects get the next free 3-digit ID (001, 002, ...).
        """
        validate_slug(slug)
        with self._lock:
            if slug in self._projects:
                return self._projects[slug]
            next_id = self._next_id()
            record = ProjectRecord(
                id=next_id,
                slug=slug,
                created=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                environments=[],
                description=description,
            )
            self._projects[slug] = record
            self._dirty = True
            return record

    def add_environment(self, slug: str, env: str) -> ProjectRecord:
        validate_slug(slug)
        validate_environment(env)
        with self._lock:
            record = self._projects.get(slug)
            if record is None:
                raise RegistryError(f"Project {slug!r} not registered.")
            if env not in record.environments:
                record.environments.append(env)
                record.environments.sort()
                self._dirty = True
            return record

    def rename(self, old_slug: str, new_slug: str) -> ProjectRecord:
        validate_slug(old_slug)
        validate_slug(new_slug)
        with self._lock:
            if old_slug not in self._projects:
                raise RegistryError(f"Project {old_slug!r} not registered.")
            if new_slug in self._projects:
                raise RegistryError(f"Project {new_slug!r} already exists.")
            record = self._projects.pop(old_slug)
            record.slug = new_slug
            self._projects[new_slug] = record
            self._dirty = True
            return record

    def drop(self, slug: str) -> ProjectRecord:
        """Remove the project record. **Caller is responsible for dropping
        the underlying collections** in Chroma/LanceDB — the registry
        only manages the map.
        """
        validate_slug(slug)
        with self._lock:
            if slug not in self._projects:
                raise RegistryError(f"Project {slug!r} not registered.")
            record = self._projects.pop(slug)
            self._dirty = True
            return record

    # ------------------------------------------------------------------ helpers

    def _next_id(self) -> str:
        used = {int(r.id) for r in self._projects.values()}
        for n in range(1, 1000):
            if n not in used:
                return f"{n:03d}"
        raise RegistryError(
            "Project ID space exhausted (000–999). Drop unused projects."
        )

    # ------------------------------------------------------------------ dunder

    def __contains__(self, slug: str) -> bool:
        return slug in self._projects

    def __len__(self) -> int:
        return len(self._projects)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def default_slug() -> str:
    return _DEFAULT_PROJECT_SLUG


def open_registry(persist_dir: Path) -> ProjectRegistry:
    """Open (and lazily load) the registry rooted at `persist_dir`."""
    registry = ProjectRegistry(persist_dir)
    registry.load()
    return registry
