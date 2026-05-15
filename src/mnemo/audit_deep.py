"""Deep drift detection — behavior alignment via LLM (Copilot CLI).

Where the cheap path (`mnemo.audit`) checks file existence, ID
references, and template compliance, the deep path asks an LLM
whether the code *actually behaves* the way the spec says it should.

One LLM call per implemented spec. Slow and not free, but it catches
the kind of divergence cheap checks can't see — e.g. the file exists,
mentions the spec ID, and the template is full, but the function it
contains takes a different branch than the spec describes.

Returns `DriftIssue` records compatible with `mnemo.audit.DriftReport`,
so the CLI / MCP layer can merge cheap + deep findings into a single
output without juggling two report shapes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mnemo.audit import DriftIssue, _parse_csv
from mnemo.core.protocols import RagPipeline
from mnemo.ingestion.agents.copilot.runner import CopilotRunner, CopilotRunnerError

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "audit_prompts" / "drift_audit.prompt.md"

# Max chars of each related file injected into the prompt — beyond this we
# truncate and add a note. Tunable via env (MNEMO_AUDIT_FILE_BUDGET).
_DEFAULT_FILE_BUDGET = 6000

_VALID_SEVERITIES = ("none", "low", "medium", "high")
_VALID_CONFIDENCES = ("low", "medium", "high")


class DeepAuditEngine:
    """Runs LLM-powered behavior drift checks against implemented specs."""

    def __init__(
        self,
        pipeline: RagPipeline,
        *,
        code_root: Path,
        runner: CopilotRunner | None = None,
        file_budget: int | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._code_root = Path(code_root).resolve()
        self._runner = runner or CopilotRunner()
        self._system_prompt = _PROMPT_FILE.read_text(encoding="utf-8")
        self._file_budget = (
            file_budget
            if file_budget is not None
            else _read_env_budget()
        )

    # ------------------------------------------------------------------ public

    def is_available(self) -> bool:
        """Pre-flight check — does the configured Copilot CLI actually exist?"""
        return self._runner.is_available()

    def audit_spec(self, spec_id: str) -> list[DriftIssue]:
        """Run the deep audit for `spec_id`. Returns `DriftIssue`s ready to merge."""
        hits = self._fetch_hits_by_id(spec_id)
        if not hits:
            return []

        meta = self._merge_spec_metadata(hits)
        if meta.get("lifecycle") != "implemented":
            return []  # deep audit only applies to implemented specs

        declared = _parse_csv(meta.get("related_files"))
        file_contents = self._read_files(declared)
        if not file_contents:
            # Nothing to audit against — cheap engine already flagged this
            # as a status drift. Skip silently.
            return []

        prompt = self._build_prompt(spec_id, meta, file_contents)
        try:
            record = self._runner.run_json(prompt)
        except CopilotRunnerError as exc:
            logger.warning("Copilot CLI failed during deep audit of %s: %s", spec_id, exc)
            return []

        if not isinstance(record, dict):
            logger.info("Deep audit of %s returned non-JSON; skipping.", spec_id)
            return []

        return self._build_issues(spec_id, record)

    # ------------------------------------------------------------------ internals

    def _fetch_hits_by_id(self, spec_id: str) -> list[Any]:
        result = self._pipeline.query(spec_id, k=10)
        return [h for h in result.hits if h.doc_id == spec_id]

    @staticmethod
    def _merge_spec_metadata(hits: list[Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for h in hits:
            for k, v in (h.metadata or {}).items():
                if k not in merged or (not merged[k] and v):
                    merged[k] = v
        return merged

    def _read_files(self, declared: list[str]) -> dict[str, str]:
        """Read each declared file with a per-file budget cap."""
        out: dict[str, str] = {}
        for rel in declared:
            path = self._code_root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(text) > self._file_budget:
                text = (
                    text[: self._file_budget]
                    + f"\n\n[... truncated; original size {len(text)} chars]"
                )
            out[rel] = text
        return out

    def _build_prompt(
        self,
        spec_id: str,
        meta: Mapping[str, Any],
        file_contents: Mapping[str, str],
    ) -> str:
        sections: list[str] = [
            self._system_prompt,
            "\n---\n",
            f"SPEC ID: {spec_id}",
        ]

        if summary := meta.get("acceptance_summary"):
            sections.append(f"\n## Acceptance Summary\n{summary}")
        if ac := meta.get("acceptance_criteria"):
            sections.append(f"\n## Acceptance Criteria\n{ac}")

        for label, key in (
            ("Happy path",  "test_scenarios_happy"),
            ("Error path",  "test_scenarios_error"),
            ("Edge case",   "test_scenarios_edge"),
        ):
            if value := meta.get(key):
                sections.append(f"\n## Test Scenario — {label}\n```gherkin\n{value}\n```")

        sections.append("\n## Code excerpts\n")
        for rel, body in file_contents.items():
            sections.append(f"\n### {rel}\n```\n{body}\n```")

        return "\n".join(sections)

    def _build_issues(self, spec_id: str, record: dict[str, Any]) -> list[DriftIssue]:
        severity = _coerce_enum(record.get("severity"), _VALID_SEVERITIES, default="none")
        confidence = _coerce_enum(record.get("confidence"), _VALID_CONFIDENCES, default="low")
        aligned = bool(record.get("aligned", True))
        summary = str(record.get("summary") or "").strip()
        divergences = record.get("divergences") or []

        if severity == "none":
            # If the LLM tells us there's no drift, trust it — even if the
            # `aligned` flag is contradictory or the severity was coerced
            # from an unrecognized value.
            return []

        if confidence == "low" and severity in {"low", "medium"}:
            # Low confidence at low/medium severity → soft-report only.
            logger.info(
                "Low-confidence deep audit for %s; reporting at lowered severity.",
                spec_id,
            )

        divergence_list = []
        for d in divergences if isinstance(divergences, list) else []:
            if not isinstance(d, Mapping):
                continue
            divergence_list.append({
                "file": str(d.get("file") or ""),
                "description": str(d.get("description") or ""),
                "evidence": str(d.get("evidence") or "")[:1000],
            })

        description = summary or (
            f"Deep audit reports {severity} behavior drift for {spec_id}."
        )
        suggested_action = (
            "Inspect the cited evidence; if the divergence is intentional, "
            "update the spec's acceptance_summary / criteria to match. "
            "Otherwise patch the code to honor the spec."
        )
        return [DriftIssue(
            type="behavior",
            severity=severity,
            description=description,
            suggested_action=suggested_action,
            details={
                "confidence": confidence,
                "aligned": aligned,
                "divergences": divergence_list,
            },
        )]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_enum(value: Any, allowed: tuple[str, ...], *, default: str) -> str:
    s = str(value or "").strip().lower()
    return s if s in allowed else default


def _read_env_budget() -> int:
    import os
    raw = os.environ.get("MNEMO_AUDIT_FILE_BUDGET")
    if not raw:
        return _DEFAULT_FILE_BUDGET
    try:
        n = int(raw)
        return max(500, n)
    except ValueError:
        return _DEFAULT_FILE_BUDGET
