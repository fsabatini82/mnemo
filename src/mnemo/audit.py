"""Drift detection — cheap, deterministic path.

Checks the alignment between spec records (what the team says the code
does or will do) and the code on disk (what the code really is). The
cheap path uses three signals, all computable in milliseconds without
LLM calls:

- **Status drift** — spec `lifecycle=implemented` but the files it
  claims to live in are missing from the working tree.
- **Coverage drift** — a spec ID is referenced by code files the spec
  doesn't list (under-claiming), or the spec lists files that don't
  actually mention the spec ID (over-claiming / stale).
- **Template drift** — spec is `partial` or `non-compliant` against
  the canonical template (computed at ingest time, surfaced here as a
  drift signal).

The deep path (behavior drift: spec says A, code does B) is F5 and
requires an LLM call per implemented spec. Not in this module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mnemo.core.protocols import RagPipeline

logger = logging.getLogger(__name__)


Severity = str  # "high" | "medium" | "low" | "none"


@dataclass(slots=True)
class DriftIssue:
    """A single drift finding for one spec."""

    type: str  # "status" | "coverage_over" | "coverage_under" | "template"
    severity: Severity
    description: str
    suggested_action: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DriftReport:
    """Aggregated drift state for one spec."""

    spec_id: str
    kind: str
    lifecycle: str
    template_compliance: str
    severity: Severity  # the worst severity across `issues`
    issues: list[DriftIssue] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "kind": self.kind,
            "lifecycle": self.lifecycle,
            "template_compliance": self.template_compliance,
            "severity": self.severity,
            "has_drift": self.has_drift,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# AuditEngine
# ---------------------------------------------------------------------------


class AuditEngine:
    """Runs the cheap drift checks against the active project's specs.

    The engine retrieves spec records from the `RagPipeline`, then
    applies one or more `DriftCheck`s per record. Aggregation logic is
    here; each check is a small pure function.
    """

    def __init__(self, pipeline: RagPipeline, *, code_root: Path) -> None:
        self._pipeline = pipeline
        self._code_root = Path(code_root).resolve()

    # ------------------------------------------------------------------ entry

    def audit_spec(self, spec_id: str) -> DriftReport | None:
        """Audit a single spec by ID. Returns None if the spec is unknown."""
        hits = self._fetch_hits_by_id(spec_id)
        if not hits:
            return None
        meta = self._merge_spec_metadata(hits)
        return self._build_report(spec_id, meta)

    def audit_all(self, *, lifecycle: str | None = None) -> list[DriftReport]:
        """Audit every spec (optionally filtered by lifecycle)."""
        result = self._pipeline.query("*", k=10_000)
        # Group hits by spec_id (multiple chunks per spec).
        by_spec: dict[str, list[Any]] = {}
        for h in result.hits:
            sid = h.doc_id or _spec_id_from_chunk(h.chunk_id)
            if not sid:
                continue
            by_spec.setdefault(sid, []).append(h)

        reports: list[DriftReport] = []
        for spec_id, hits in by_spec.items():
            meta = self._merge_spec_metadata(hits)
            if lifecycle is not None and meta.get("lifecycle") != lifecycle:
                continue
            reports.append(self._build_report(spec_id, meta))
        return reports

    # ------------------------------------------------------------------ checks

    def _build_report(self, spec_id: str, meta: dict[str, Any]) -> DriftReport:
        issues: list[DriftIssue] = []
        issues.extend(self._check_status_drift(spec_id, meta))
        issues.extend(self._check_coverage_drift(spec_id, meta))
        issues.extend(self._check_template_drift(spec_id, meta))

        severity = _worst_severity(issues) if issues else "none"
        return DriftReport(
            spec_id=spec_id,
            kind=str(meta.get("kind") or ""),
            lifecycle=str(meta.get("lifecycle") or ""),
            template_compliance=str(meta.get("template_compliance") or "n/a"),
            severity=severity,
            issues=issues,
        )

    def _check_status_drift(self, spec_id: str, meta: dict[str, Any]) -> list[DriftIssue]:
        """Lifecycle=implemented but related_files missing on disk."""
        if meta.get("lifecycle") != "implemented":
            return []
        declared = _parse_csv(meta.get("related_files"))
        if not declared:
            # Implemented without any related_files declared — soft signal.
            return [DriftIssue(
                type="status",
                severity="low",
                description=(
                    f"Spec {spec_id} is `implemented` but declares no "
                    "related_files. Add at least one file reference for "
                    "traceability."
                ),
                suggested_action="Set `related_files: [...]` in the spec frontmatter.",
            )]

        missing = [f for f in declared if not (self._code_root / f).is_file()]
        if not missing:
            return []
        return [DriftIssue(
            type="status",
            severity="high",
            description=(
                f"Spec {spec_id} is `implemented` but {len(missing)} of "
                f"{len(declared)} declared files are missing from the working tree."
            ),
            suggested_action=(
                "Either ship the implementation, downgrade lifecycle to "
                "`proposed`, or update `related_files` to match the current code."
            ),
            details={"declared": declared, "missing": missing},
        )]

    def _check_coverage_drift(self, spec_id: str, meta: dict[str, Any]) -> list[DriftIssue]:
        """Look for over-claim (declared files that don't mention spec_id)
        and under-claim (files that reference spec_id but aren't declared).
        """
        if meta.get("lifecycle") != "implemented":
            return []

        declared = _parse_csv(meta.get("related_files"))
        existing = [f for f in declared if (self._code_root / f).is_file()]

        # Over-claim: declared files that exist but don't reference the spec ID.
        over = [
            f for f in existing
            if not _file_references_id(self._code_root / f, spec_id)
        ]

        # Under-claim: scan the codebase for files mentioning spec_id that
        # aren't in `declared`. Keep the scan cheap by limiting to text-like
        # extensions and skipping common noise dirs.
        all_refs = _scan_codebase_for_id(self._code_root, spec_id)
        under = sorted(set(all_refs) - set(declared))

        issues: list[DriftIssue] = []
        if over:
            issues.append(DriftIssue(
                type="coverage_over",
                severity="medium",
                description=(
                    f"{len(over)} declared file(s) for {spec_id} don't mention "
                    "its ID anywhere in their content (potential over-claim)."
                ),
                suggested_action=(
                    "Verify the file actually implements the spec, or remove "
                    "it from `related_files`."
                ),
                details={"over_claimed": over},
            ))
        if under:
            issues.append(DriftIssue(
                type="coverage_under",
                severity="medium",
                description=(
                    f"{len(under)} file(s) mention {spec_id} but aren't "
                    "listed in `related_files` (potential under-claim)."
                ),
                suggested_action=(
                    "Add the relevant files to `related_files` for traceability."
                ),
                details={"under_claimed": under},
            ))
        return issues

    def _check_template_drift(self, spec_id: str, meta: dict[str, Any]) -> list[DriftIssue]:
        compliance = str(meta.get("template_compliance") or "n/a")
        if compliance in {"full", "n/a"}:
            return []

        severity = "low" if compliance == "partial" else "medium"
        return [DriftIssue(
            type="template",
            severity=severity,
            description=(
                f"Spec {spec_id} is `{compliance}` against the canonical "
                f"`{meta.get('kind', 'spec')}` template."
            ),
            suggested_action=(
                f"Run `mnemo-admin new-spec {meta.get('kind') or 'story'} "
                "--id {spec_id} --title ...` to inspect the canonical "
                "structure, then backfill the missing sections."
            ),
        )]

    # ------------------------------------------------------------------ helpers

    def _fetch_hits_by_id(self, spec_id: str) -> list[Any]:
        result = self._pipeline.query(spec_id, k=10)
        return [h for h in result.hits if h.doc_id == spec_id]

    @staticmethod
    def _merge_spec_metadata(hits: list[Any]) -> dict[str, Any]:
        """Merge metadata across all chunks of a spec.

        We prefer non-empty values from any chunk. Chunks share most
        metadata anyway (per-spec frontmatter), so this is mostly a
        defensive merge.
        """
        merged: dict[str, Any] = {}
        for h in hits:
            for k, v in (h.metadata or {}).items():
                if k not in merged or (not merged[k] and v):
                    merged[k] = v
        return merged


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_csv(raw: Any) -> list[str]:
    """Split a comma-separated metadata string into a list of strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _spec_id_from_chunk(chunk_id: str) -> str:
    """Extract the spec id from a chunk id of the form `<id>::chunk-NNNN`."""
    if "::chunk-" in chunk_id:
        return chunk_id.split("::chunk-")[0]
    return chunk_id


_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _worst_severity(issues: list[DriftIssue]) -> Severity:
    if not issues:
        return "none"
    return max(issues, key=lambda i: _SEVERITY_RANK.get(i.severity, 0)).severity


_SCAN_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
    ".cs", ".cpp", ".c", ".h", ".hpp", ".rb", ".php", ".kt", ".swift",
    ".sql", ".yaml", ".yml", ".json", ".toml", ".md",
}

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
    ".cache", ".idea", ".vscode",
}


def _scan_codebase_for_id(root: Path, spec_id: str) -> list[str]:
    """Find files under `root` that contain `spec_id` as a token.

    Best-effort: limited to common source extensions, skips standard
    cache/build dirs. Returns POSIX-style relative paths so the result
    is comparable to declared `related_files` regardless of OS.
    """
    if not root.is_dir():
        return []
    # Word-boundary match to avoid e.g. "US-1" matching "US-10".
    pattern = re.compile(rf"\b{re.escape(spec_id)}\b")
    found: list[str] = []
    for path in _walk(root):
        if path.suffix.lower() not in _SCAN_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            found.append(rel)
    return found


def _walk(root: Path):
    """Iterator that skips common noise directories."""
    for dirpath, dirnames, filenames in _safe_walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            yield Path(dirpath) / fname


def _safe_walk(root: Path):
    import os
    for entry in os.walk(root):
        yield entry


def _file_references_id(path: Path, spec_id: str) -> bool:
    """Return True if `spec_id` appears as a token in `path`."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(re.search(rf"\b{re.escape(spec_id)}\b", text))
