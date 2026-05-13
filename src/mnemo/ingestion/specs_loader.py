"""Specs ingestion adapter.

For the lab this reads Markdown files from a local folder that mimics an
exported Wiki / ADO Wiki dump. Each file becomes one `Document` whose
metadata is enriched with the YAML frontmatter (when present).

PRODUCTION NOTE
---------------
In a real deployment, replace `iter_spec_files` with a call to:
  • Azure DevOps Wiki REST API
  • Confluence REST API
  • GitHub Wiki / a docs repo clone
  • SharePoint / OneDrive listing
The downstream pipeline does not change — only the source adapter does.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from mnemo.core.models import Document

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def iter_spec_files(source_dir: Path) -> Iterator[Path]:
    """Yield every spec file under `source_dir`.

    Replace this with an API call in production — see module docstring.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"specs source not found: {source_dir}")
    yield from sorted(source_dir.rglob("*.md"))


def parse_spec(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(raw)

    metadata.setdefault("source_file", str(path.relative_to(root)))
    metadata.setdefault("kind", _infer_kind(path, root))

    # The doc id favors the frontmatter `id` (e.g. "US-102") and falls
    # back to the relative path. Stable IDs let re-ingest behave as upsert.
    doc_id = str(metadata.get("id") or path.relative_to(root))

    # We keep the frontmatter inline in the indexed text so retrieval can
    # match on titles and tags, not just body content.
    indexed_text = _render_indexed_text(metadata, body)
    # Vector stores only accept scalar metadata values (str | int | float |
    # bool | None). YAML auto-coerces ISO dates into datetime.date and
    # comma lists into Python lists — flatten everything before handing
    # off to the store.
    return Document(id=doc_id, text=indexed_text, metadata=_flatten_metadata(metadata))


def load_specs(source_dir: Path) -> list[Document]:
    return [parse_spec(p, source_dir) for p in iter_spec_files(source_dir)]


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_raw, body = match.groups()
    metadata = yaml.safe_load(fm_raw) or {}
    if not isinstance(metadata, dict):
        return {}, raw
    return metadata, body


def _infer_kind(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    if rel.startswith("epics/"):
        return "epic"
    if rel.startswith("stories/"):
        return "story"
    if rel.startswith("adrs/"):
        return "adr"
    return "spec"


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata values to scalars accepted by Chroma / LanceDB.

    - None / bool / int / float / str → passthrough
    - list / tuple → comma-joined string
    - everything else (date, datetime, dict, custom objects) → str()
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
    """Compose the text that actually gets embedded.

    Including title and key metadata in the embedded text materially
    improves semantic recall on identifier-shaped queries (e.g. "US-102").
    """
    header_lines = []
    if title := metadata.get("title"):
        header_lines.append(f"# {metadata.get('id', '')} — {title}".strip())
    if epic := metadata.get("epic"):
        header_lines.append(f"Epic: {epic}")
    if rb := metadata.get("related_bugs"):
        header_lines.append(f"Related bugs: {', '.join(rb)}")
    if rs := metadata.get("related_adrs"):
        header_lines.append(f"Related ADRs: {', '.join(rs)}")
    header = "\n".join(header_lines)
    return f"{header}\n\n{body}".strip() if header else body
