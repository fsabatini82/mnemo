"""Structural interfaces (PEP 544) for the swappable components.

Using `Protocol` rather than ABC lets third-party adapters (LlamaIndex,
LanceDB, …) satisfy these contracts without inheritance — they just need
to expose the right methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from mnemo.core.models import Chunk, Document, Hit, QueryResult

# Type alias for metadata equality filters pushed down to the store.
# Format: `{"key": "value"}` for equality. Stores translate this to
# their native filter syntax (Chroma `where=`, LanceDB SQL WHERE).
MetadataFilter = Mapping[str, Any]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists chunks and retrieves them by vector similarity."""

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[Hit]: ...


@runtime_checkable
class SupportsHybridSearch(Protocol):
    """Optional capability: combined vector + keyword retrieval.

    A `VectorStore` may also satisfy this protocol when it supports a
    fused dense+sparse query (e.g. LanceDB with FTS index).
    """

    def hybrid_search(
        self,
        embedding: Sequence[float],
        query_text: str,
        *,
        k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[Hit]: ...


@runtime_checkable
class RagPipeline(Protocol):
    """High-level ingestion + retrieval facade exposed to the MCP layer."""

    def ingest(self, documents: Sequence[Document]) -> None: ...

    def query(
        self,
        question: str,
        *,
        k: int = 5,
        where: MetadataFilter | None = None,
    ) -> QueryResult: ...


def matches_where(metadata: Mapping[str, Any], where: MetadataFilter) -> bool:
    """Client-side equality filter shared by adapters that can't push down.

    Supports plain `{"key": "value"}` equality and Chroma-style operators:

    - `{"$in": [...]}`     value is in the list
    - `{"$ne": x}`         not equal to x

    Other operators fall through and treat the expected dict as a
    literal mismatch (no false positives).
    """
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            if "$in" in expected:
                if actual not in expected["$in"]:
                    return False
                continue
            if "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
                continue
            return False  # unknown operator → don't risk a wrong match
        if actual != expected:
            return False
    return True
