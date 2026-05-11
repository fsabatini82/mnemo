"""Structural interfaces (PEP 544) for the swappable components.

Using `Protocol` rather than ABC lets third-party adapters (LlamaIndex,
LanceDB, …) satisfy these contracts without inheritance — they just need
to expose the right methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mnemo.core.models import Chunk, Document, Hit, QueryResult


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

    def search(self, embedding: Sequence[float], *, k: int = 5) -> list[Hit]: ...


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
    ) -> list[Hit]: ...


@runtime_checkable
class RagPipeline(Protocol):
    """High-level ingestion + retrieval facade exposed to the MCP layer."""

    def ingest(self, documents: Sequence[Document]) -> None: ...

    def query(self, question: str, *, k: int = 5) -> QueryResult: ...
