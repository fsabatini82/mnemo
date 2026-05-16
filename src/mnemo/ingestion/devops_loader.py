"""DevOps ingestion adapter.

Reads Markdown files from a folder mirror produced by an Azure DevOps
export. Each file is one work item (Feature, PBI, or open Bug); Tasks
are excluded by the exporter, resolved Bugs go to the `bugs` stream.

Frontmatter is YAML with DevOps-shaped fields (id, work_item_type,
state, parent_id, area_path, ...); body is markdown with optional
`## Description`, `## Acceptance Criteria`, `## Repro Steps`,
`## Discussion` sections. The exporter is responsible for HTML→markdown
conversion of DevOps source fields.

State vocabulary is faithful to DevOps verbatim (including custom
process-template values like "Validato BU", "Progettato"). No
normalization to Mnemo's `lifecycle` vocabulary — that lives in the
specs stream and is owned by humans.

PRODUCTION NOTE
---------------
In a real deployment, replace `iter_devops_files` with a streaming
call to the Azure DevOps Work Items REST API. The downstream pipeline
does not change — only the source adapter does.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from mnemo.core.models import Document

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def iter_devops_files(source_dir: Path) -> Iterator[Path]:
    """Yield every devops mirror file under `source_dir`.

    Replace this with an API call in production — see module docstring.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"devops source not found: {source_dir}")
    yield from sorted(source_dir.rglob("*.md"))


def parse_devops(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(raw)

    metadata.setdefault("source_file", str(path.relative_to(root)))
    # Doc id = frontmatter `id` (e.g. "WI-396679"); falls back to
    # the relative path so the loader stays usable on hand-crafted
    # files without an id.
    doc_id = str(metadata.get("id") or path.relative_to(root))

    indexed_text = _render_indexed_text(metadata, body)
    return Document(id=doc_id, text=indexed_text, metadata=_flatten_metadata(metadata))


def load_devops(source_dir: Path) -> list[Document]:
    return [parse_devops(p, source_dir) for p in iter_devops_files(source_dir)]


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_raw, body = match.groups()
    metadata = yaml.safe_load(fm_raw) or {}
    if not isinstance(metadata, dict):
        return {}, raw
    return metadata, body


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata values to scalars accepted by Chroma / LanceDB.

    - None / bool / int / float / str → passthrough
    - list / tuple → comma-joined string
    - everything else (date, datetime, dict, ...) → str()
    """
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            flat[k] = v
        elif isinstance(v, (list, tuple)):
            flat[k] = ", ".join(str(x) for x in v)
        else:
            flat[k] = str(v)
    return flat


def _render_indexed_text(metadata: dict[str, Any], body: str) -> str:
    """Compose the text that gets embedded.

    The header surfaces identifier-shaped fields so semantic recall
    works for queries like "WI-396679" or "PBI sullo SPID" — the
    embedding model sees the id, title, type and state alongside the
    body, not just the prose.
    """
    header_lines: list[str] = []
    wi_id = metadata.get("id", "")
    if title := metadata.get("title"):
        header_lines.append(f"# {wi_id} — {title}".strip(" —"))
    if wit := metadata.get("work_item_type"):
        state = metadata.get("state", "")
        line = f"Type: {wit}"
        if state:
            line += f" | State: {state}"
        header_lines.append(line)
    if parent_title := metadata.get("parent_title"):
        header_lines.append(f"Parent: {parent_title}")
    if assigned := metadata.get("assigned_to"):
        header_lines.append(f"Assigned to: {assigned}")
    header = "\n".join(header_lines)
    return f"{header}\n\n{body}".strip() if header else body
