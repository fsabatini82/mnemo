"""Protocol for LLM-powered ingestion agents.

An ingestion agent is the LLM-driven alternative to the deterministic
`specs_loader` / `bugs_loader`. Same output (`list[Document]`),
different internals: rather than parsing files mechanically, the agent
asks an LLM to extract structured metadata, classify items as
indexable vs noise, and enrich cross-references.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from mnemo.core.models import Document


@runtime_checkable
class IngestionAgent(Protocol):
    """Read items from `source_dir`, return canonical Documents.

    Implementations may skip items the agent decides are not worth
    indexing (e.g. trivial typo bugs, doc-only changes) — this is the
    main value-add over the deterministic loader.
    """

    def ingest(self, source_dir: Path) -> list[Document]: ...
