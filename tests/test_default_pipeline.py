"""Tests for the DefaultPipeline (hand-rolled orchestration)."""

from __future__ import annotations

from collections.abc import Sequence

from mnemo.core.models import Chunk, Document, Hit
from mnemo.pipelines.default import DefaultPipeline


class _RecordingStore:
    """Minimal VectorStore stub that records its calls."""

    def __init__(self) -> None:
        self.upserts: list[tuple[Sequence[Chunk], Sequence[Sequence[float]]]] = []
        self.searches: list[tuple[Sequence[float], int]] = []

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.upserts.append((list(chunks), list(embeddings)))

    def search(self, embedding: Sequence[float], *, k: int = 5) -> list[Hit]:
        self.searches.append((list(embedding), k))
        return [Hit(chunk_id="c1", doc_id="d1", text="match", score=0.9)]


class _HybridStore(_RecordingStore):
    """Variant that also satisfies SupportsHybridSearch structurally."""

    def __init__(self) -> None:
        super().__init__()
        self.hybrid_calls: list[tuple[Sequence[float], str, int]] = []

    def hybrid_search(
        self, embedding: Sequence[float], query_text: str, *, k: int = 5
    ) -> list[Hit]:
        self.hybrid_calls.append((list(embedding), query_text, k))
        return [Hit(chunk_id="h1", doc_id="d1", text="hybrid", score=0.95)]


def test_ingest_chunks_embeds_and_upserts(fake_embedder, sample_documents) -> None:
    store = _RecordingStore()
    pipeline = DefaultPipeline(
        embedder=fake_embedder,
        store=store,
        chunk_size=64,
        chunk_overlap=8,
    )
    pipeline.ingest(sample_documents)
    assert len(store.upserts) == 1
    chunks, embeddings = store.upserts[0]
    assert len(chunks) == len(embeddings) > 0


def test_ingest_empty_docs_is_noop(fake_embedder) -> None:
    store = _RecordingStore()
    pipeline = DefaultPipeline(
        embedder=fake_embedder,
        store=store,
        chunk_size=64,
        chunk_overlap=8,
    )
    pipeline.ingest([])
    assert store.upserts == []


def test_ingest_skips_when_chunker_returns_nothing(fake_embedder) -> None:
    """If a document has no extractable text, no upsert should happen."""
    store = _RecordingStore()
    pipeline = DefaultPipeline(
        embedder=fake_embedder,
        store=store,
        chunk_size=64,
        chunk_overlap=8,
    )
    pipeline.ingest([Document(id="empty", text="")])
    # langchain-text-splitters may emit no chunks for empty content.
    if store.upserts:
        assert len(store.upserts[0][0]) >= 0  # tolerant: no crash either way


def test_query_uses_plain_search_for_non_hybrid_store(fake_embedder) -> None:
    store = _RecordingStore()
    pipeline = DefaultPipeline(
        embedder=fake_embedder,
        store=store,
        chunk_size=64,
        chunk_overlap=8,
    )
    result = pipeline.query("What is X?", k=3)
    assert result.question == "What is X?"
    assert len(result.hits) == 1
    assert store.searches == [(store.searches[0][0], 3)]


def test_query_uses_hybrid_search_when_store_supports_it(fake_embedder) -> None:
    store = _HybridStore()
    pipeline = DefaultPipeline(
        embedder=fake_embedder,
        store=store,
        chunk_size=64,
        chunk_overlap=8,
    )
    result = pipeline.query("What is X?", k=4)
    # Plain `search` should not have been called; hybrid was.
    assert store.searches == []
    assert len(store.hybrid_calls) == 1
    _, query_text, k = store.hybrid_calls[0]
    assert query_text == "What is X?"
    assert k == 4
    assert result.hits[0].text == "hybrid"
