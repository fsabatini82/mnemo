"""Hand-rolled RAG pipeline — the explicit, didactic path.

Composes `Embedder` + `VectorStore` (+ optional `SupportsHybridSearch`)
into the standard ingest/query flow, without any framework magic.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnemo.chunking import chunk_documents
from mnemo.core.models import Document, QueryResult
from mnemo.core.protocols import Embedder, SupportsHybridSearch, VectorStore


class DefaultPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, documents: Sequence[Document]) -> None:
        chunks = chunk_documents(
            documents,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        if not chunks:
            return
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._store.upsert(chunks, embeddings)

    def query(self, question: str, *, k: int = 5) -> QueryResult:
        embedding = self._embedder.embed([question])[0]
        if isinstance(self._store, SupportsHybridSearch):
            hits = self._store.hybrid_search(embedding, question, k=k)
        else:
            hits = self._store.search(embedding, k=k)
        return QueryResult(question=question, hits=hits)
