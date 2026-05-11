"""Bug-memory ingestion adapter.

For the lab this reads JSON files from a local folder that mimics an
exported Azure DevOps work-item dump. Each file is one resolved bug.

PRODUCTION NOTE
---------------
In a real deployment, replace `iter_bug_files` with a call to:
  • Azure DevOps Work Items REST API (state=Resolved, type=Bug)
  • Jira REST API (resolution != Unresolved)
  • GitHub Issues API (state=closed, label=bug)
  • Linear / Sentry / Bugsnag exports
The downstream pipeline does not change — only the source adapter does.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mnemo.core.models import Document

# Fields that contribute to the embedded text. Order matters for retrieval
# weighting: the title carries the strongest signal, the fix the weakest.
_EMBEDDED_FIELDS = ("title", "symptom", "root_cause", "fix_summary")


def iter_bug_files(source_dir: Path) -> Iterator[Path]:
    """Yield every bug file under `source_dir`.

    Replace this with an API call in production — see module docstring.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"bugs source not found: {source_dir}")
    yield from sorted(source_dir.rglob("*.json"))


def parse_bug(path: Path) -> Document:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    bug_id = str(record.get("id") or path.stem)

    text = _render_indexed_text(bug_id, record)

    # Flatten useful metadata. Lists are joined to strings because some
    # vector stores (e.g. Chroma) only accept scalar metadata values.
    metadata: dict[str, Any] = {
        "kind": "bug",
        "bug_id": bug_id,
        "title": record.get("title", ""),
        "severity": record.get("severity", ""),
        "epic": record.get("epic", ""),
        "status": record.get("status", ""),
        "related_spec": record.get("related_spec", ""),
        "related_adr": record.get("related_adr", ""),
        "related_pr": record.get("related_pr", ""),
        "files_touched": ", ".join(record.get("files_touched", []) or []),
        "pattern_tags": ", ".join(record.get("pattern_tags", []) or []),
        "resolved": record.get("resolved", ""),
        "resolved_by": record.get("resolved_by", ""),
    }
    return Document(id=bug_id, text=text, metadata=metadata)


def load_bugs(source_dir: Path) -> list[Document]:
    return [parse_bug(p) for p in iter_bug_files(source_dir)]


def _render_indexed_text(bug_id: str, record: dict[str, Any]) -> str:
    parts = [f"# {bug_id} — {record.get('title', '').strip()}"]
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
