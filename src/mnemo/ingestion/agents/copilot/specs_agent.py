"""LLM-powered specs ingestion agent (GitHub Copilot CLI backend).

Reads markdown spec files from a folder, asks the Copilot CLI to
extract structured metadata + classify as indexable/noise, builds
canonical `Document` objects. Skips items the agent marks as noise.

Headless and schedulable (Task Scheduler, cron). No interactive
prompts; auth is whatever the Copilot CLI already has cached.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mnemo.core.models import Document
from mnemo.ingestion.agents.copilot.runner import CopilotRunner, CopilotRunnerError

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "specs.prompt.md"


class CopilotSpecsAgent:
    """Satisfies `IngestionAgent` Protocol structurally."""

    def __init__(self, runner: CopilotRunner | None = None) -> None:
        self._runner = runner or CopilotRunner()
        self._system_prompt = _PROMPT_FILE.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ public

    def ingest(self, source_dir: Path) -> list[Document]:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"specs source not found: {source_dir}")

        documents: list[Document] = []
        for path in _iter_markdown(source_dir):
            try:
                doc = self._process_one(path, source_dir)
            except CopilotRunnerError as exc:
                logger.warning("Copilot CLI failed on %s: %s — skipping", path, exc)
                continue
            if doc is not None:
                documents.append(doc)
        return documents

    # ------------------------------------------------------------------ internals

    def _process_one(self, path: Path, root: Path) -> Document | None:
        raw = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()

        prompt = self._build_prompt(rel, raw)
        record = self._runner.run_json(prompt)
        if not isinstance(record, dict):
            logger.info("No JSON record from agent for %s — skipping", rel)
            return None

        if not record.get("indexable", False):
            logger.info("Agent classified %s as noise — skipping", rel)
            return None

        doc_id = str(record.get("id") or rel)
        metadata: dict[str, Any] = {
            "source_file": rel,
            "kind": str(record.get("kind") or "spec"),
            "title": record.get("title") or "",
            "epic": record.get("epic") or "",
            "status": record.get("status") or "",
            "related_bugs": ", ".join(record.get("related_bugs") or []),
            "related_adrs": ", ".join(record.get("related_adrs") or []),
            "related_files": ", ".join(record.get("related_files") or []),
            "extracted_by": "copilot-cli",
        }
        body = record.get("body") or raw  # fall back to original if model omitted body
        text = self._render_indexed_text(doc_id, record, body)
        return Document(id=doc_id, text=text, metadata=metadata)

    def _build_prompt(self, rel_path: str, body: str) -> str:
        return (
            f"{self._system_prompt}\n\n"
            f"---\n\n"
            f"FILE: {rel_path}\n\n"
            f"CONTENT:\n```markdown\n{body}\n```\n"
        )

    @staticmethod
    def _render_indexed_text(doc_id: str, record: dict[str, Any], body: str) -> str:
        header_lines: list[str] = []
        if title := record.get("title"):
            header_lines.append(f"# {doc_id} — {title}")
        if epic := record.get("epic"):
            header_lines.append(f"Epic: {epic}")
        if rb := record.get("related_bugs"):
            header_lines.append(f"Related bugs: {', '.join(rb)}")
        if ra := record.get("related_adrs"):
            header_lines.append(f"Related ADRs: {', '.join(ra)}")
        header = "\n".join(header_lines)
        return f"{header}\n\n{body}".strip() if header else body


def _iter_markdown(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.md"))
