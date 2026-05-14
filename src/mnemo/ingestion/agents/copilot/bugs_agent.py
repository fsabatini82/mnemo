"""LLM-powered bugs ingestion agent (GitHub Copilot CLI backend).

Reads JSON work-item dumps from a folder, asks the Copilot CLI to
extract a normalized lesson-shaped record and classify each bug as
indexable (carries a reusable lesson) or noise (typo fix, doc-only,
cosmetic). Skips the noise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mnemo.core.models import Document
from mnemo.ingestion.agents.copilot.runner import CopilotRunner, CopilotRunnerError

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "bugs.prompt.md"
_EMBEDDED_FIELDS = ("title", "symptom", "root_cause", "fix_summary")


class CopilotBugsAgent:
    """Satisfies `IngestionAgent` Protocol structurally."""

    def __init__(self, runner: CopilotRunner | None = None) -> None:
        self._runner = runner or CopilotRunner()
        self._system_prompt = _PROMPT_FILE.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ public

    def ingest(self, source_dir: Path) -> list[Document]:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"bugs source not found: {source_dir}")

        documents: list[Document] = []
        for path in _iter_json(source_dir):
            try:
                doc = self._process_one(path)
            except CopilotRunnerError as exc:
                logger.warning("Copilot CLI failed on %s: %s — skipping", path, exc)
                continue
            if doc is not None:
                documents.append(doc)
        return documents

    # ------------------------------------------------------------------ internals

    def _process_one(self, path: Path) -> Document | None:
        try:
            raw = path.read_text(encoding="utf-8")
            json.loads(raw)  # validate it's at least JSON before paying for an LLM call
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable bug file %s: %s", path, exc)
            return None

        prompt = self._build_prompt(path.name, raw)
        record = self._runner.run_json(prompt)
        if not isinstance(record, dict):
            logger.info("No JSON record from agent for %s — skipping", path.name)
            return None

        if not record.get("indexable", False):
            logger.info("Agent classified %s as noise — skipping", path.name)
            return None

        bug_id = str(record.get("id") or path.stem)
        metadata: dict[str, Any] = {
            "kind": "bug",
            "bug_id": bug_id,
            "title": record.get("title") or "",
            "severity": record.get("severity") or "",
            "epic": record.get("epic") or "",
            "status": record.get("status") or "",
            "related_spec": record.get("related_spec") or "",
            "related_pr": record.get("related_pr") or "",
            "files_touched": ", ".join(record.get("files_touched") or []),
            "pattern_tags": ", ".join(record.get("pattern_tags") or []),
            "resolved": record.get("resolved") or "",
            "extracted_by": "copilot-cli",
        }
        text = self._render_indexed_text(bug_id, record)
        return Document(id=bug_id, text=text, metadata=metadata)

    def _build_prompt(self, filename: str, raw_json: str) -> str:
        return (
            f"{self._system_prompt}\n\n"
            f"---\n\n"
            f"SOURCE FILE: {filename}\n\n"
            f"WORK ITEM JSON:\n```json\n{raw_json}\n```\n"
        )

    @staticmethod
    def _render_indexed_text(bug_id: str, record: dict[str, Any]) -> str:
        parts: list[str] = [f"# {bug_id} — {record.get('title', '').strip()}"]
        for field in _EMBEDDED_FIELDS:
            if field == "title":
                continue
            value = record.get(field, "")
            if value:
                parts.append(f"## {field.replace('_', ' ').title()}\n{value}")
        if tags := record.get("pattern_tags"):
            parts.append("## Pattern Tags\n" + ", ".join(tags))
        if files := record.get("files_touched"):
            parts.append("## Files Touched\n" + "\n".join(f"- {f}" for f in files))
        return "\n\n".join(parts)


def _iter_json(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.json"))
