"""Mnemo MCP server — exposes organizational memory to IDE agents.

Two knowledge axes:
  • specs       — what to build (user stories, ADRs, epics)
  • bug_memory  — what went wrong before (resolved bugs with root causes)

Tools are intentionally narrow and composable so an agent can chain them
during multi-step reasoning (e.g. fetch a spec → look for related bugs →
generate code).
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from mnemo.config import load_settings
from mnemo.factory import MnemoSystem, build_system

logger = logging.getLogger(__name__)

_settings = load_settings()
_system: MnemoSystem = build_system(_settings)

mcp = FastMCP("mnemo")


# ---------------------------------------------------------------------------
# Specs side — "what to build"
# ---------------------------------------------------------------------------


@mcp.tool()
def query_specs(question: str, k: int | None = None) -> dict[str, Any]:
    """Semantic search across the project specs (user stories, ADRs, epics).

    Use this when implementing a feature: it surfaces the acceptance
    criteria, related ADRs, and any cross-references the spec carries.

    Args:
        question: Natural-language question or spec identifier to look up.
        k: Optional override for the number of chunks to return.
    """
    result = _system.specs.query(question, k=k or _settings.top_k)
    return {
        "axis": "specs",
        "question": result.question,
        "hits": [_hit_to_dict(h) for h in result.hits],
    }


@mcp.tool()
def get_spec(spec_id: str) -> dict[str, Any]:
    """Retrieve a spec by its identifier (e.g. "US-102", "ADR-002").

    Equivalent to a focused query on the spec ID — useful when the agent
    already knows which item it needs and wants the full content.
    """
    return query_specs(spec_id, k=10)


# ---------------------------------------------------------------------------
# Bug memory side — "what went wrong before"
# ---------------------------------------------------------------------------


@mcp.tool()
def query_bugs(symptom: str, k: int | None = None) -> dict[str, Any]:
    """Semantic search across the resolved bug history.

    Use this when debugging or before modifying a sensitive area: it
    surfaces past bugs whose symptom or root cause resembles the input,
    along with the fix that worked.

    Args:
        symptom: Description of the issue, error message, or area of concern.
        k: Optional override for the number of bugs to return.
    """
    result = _system.bugs.query(symptom, k=k or _settings.top_k)
    return {
        "axis": "bug_memory",
        "question": result.question,
        "hits": [_hit_to_dict(h) for h in result.hits],
    }


@mcp.tool()
def get_bug(bug_id: str) -> dict[str, Any]:
    """Retrieve a bug by its identifier (e.g. "BUG-503").

    Returns the full record with symptom, root cause, fix summary, files
    touched, and pattern tags — everything an agent needs to apply the
    same fix shape to a similar new problem.
    """
    return query_bugs(bug_id, k=3)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@mcp.tool()
def mnemo_info() -> dict[str, Any]:
    """Return the active configuration — useful for sanity checks in demos."""
    return {
        "store": _settings.store,
        "pipeline": _settings.pipeline,
        "embed_model": _settings.embed_model,
        "collections": {
            "specs": _settings.specs_collection,
            "bugs": _settings.bugs_collection,
        },
        "sources": {
            "specs_dir": str(_settings.specs_source_dir),
            "bugs_dir": str(_settings.bugs_source_dir),
        },
        "chunking": {
            "size": _settings.chunk_size,
            "overlap": _settings.chunk_overlap,
        },
        "top_k": _settings.top_k,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit_to_dict(h: Any) -> dict[str, Any]:
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "score": h.score,
        "text": h.text,
        "metadata": h.metadata,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
