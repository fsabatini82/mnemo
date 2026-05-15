"""Tests for the Chroma-backed vector store.

Uses a real Chroma PersistentClient in `tmp_path` — Chroma is embedded,
no external service needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemo.core.models import Chunk
from mnemo.stores.chroma_store import ChromaStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(persist_dir=tmp_path / "data", collection="test")


def _vec(seed: float, dim: int = 4) -> list[float]:
    return [seed + i * 0.01 for i in range(dim)]


def test_upsert_and_search_round_trip(store: ChromaStore) -> None:
    chunks = [
        Chunk(id="c-1", doc_id="d-1", text="alpha", metadata={"k": "v1"}),
        Chunk(id="c-2", doc_id="d-1", text="beta", metadata={"k": "v2"}),
    ]
    embeddings = [_vec(0.1), _vec(0.9)]
    store.upsert(chunks, embeddings)

    hits = store.search(_vec(0.1), k=2)
    assert len(hits) >= 1
    # Best hit should be c-1 (nearest to its own embedding).
    assert hits[0].chunk_id == "c-1"
    assert hits[0].doc_id == "d-1"
    assert hits[0].text == "alpha"
    # Metadata preserved (minus doc_id which is hoisted into the Hit struct).
    assert hits[0].metadata == {"k": "v1"}


def test_upsert_length_mismatch_raises(store: ChromaStore) -> None:
    with pytest.raises(ValueError):
        store.upsert(
            [Chunk(id="c", doc_id="d", text="t")],
            [_vec(0.1), _vec(0.2)],  # 1 chunk vs 2 embeddings
        )


def test_upsert_empty_is_noop(store: ChromaStore) -> None:
    store.upsert([], [])
    # No exception; subsequent search returns empty.
    assert store.search(_vec(0.1), k=3) == []


def test_search_returns_similarity_score(store: ChromaStore) -> None:
    store.upsert(
        [Chunk(id="c-1", doc_id="d-1", text="alpha")],
        [_vec(0.5)],
    )
    hits = store.search(_vec(0.5), k=1)
    assert len(hits) == 1
    # Cosine distance ~0 → similarity ~1. Allow generous slack.
    assert hits[0].score > 0.5


def test_upsert_overwrites_existing_id(store: ChromaStore, tmp_path: Path) -> None:
    chunk_v1 = Chunk(id="c-1", doc_id="d-1", text="version 1")
    chunk_v2 = Chunk(id="c-1", doc_id="d-1", text="version 2")
    store.upsert([chunk_v1], [_vec(0.1)])
    store.upsert([chunk_v2], [_vec(0.1)])
    hits = store.search(_vec(0.1), k=5)
    # Only one record after the second upsert — same id replaced.
    matching = [h for h in hits if h.chunk_id == "c-1"]
    assert len(matching) == 1
    assert matching[0].text == "version 2"


def test_store_persists_across_instances(tmp_path: Path) -> None:
    persist = tmp_path / "data"
    s1 = ChromaStore(persist_dir=persist, collection="persist-test")
    s1.upsert(
        [Chunk(id="c-1", doc_id="d-1", text="hello")],
        [_vec(0.3)],
    )

    s2 = ChromaStore(persist_dir=persist, collection="persist-test")
    hits = s2.search(_vec(0.3), k=1)
    assert len(hits) == 1
    assert hits[0].text == "hello"


# ---------------------------------------------------------------------------
# Metadata where-filter (F2)
# ---------------------------------------------------------------------------


def test_search_with_where_filters_by_metadata(store: ChromaStore) -> None:
    """Push-down `where` filter must respect metadata equality."""
    chunks = [
        Chunk(id="c-impl", doc_id="d-1", text="implemented body",
              metadata={"lifecycle": "implemented"}),
        Chunk(id="c-prop", doc_id="d-2", text="proposed body",
              metadata={"lifecycle": "proposed"}),
        Chunk(id="c-asis", doc_id="d-3", text="as-is body",
              metadata={"lifecycle": "as-is"}),
    ]
    store.upsert(chunks, [_vec(0.1), _vec(0.2), _vec(0.3)])

    hits = store.search(_vec(0.1), k=5, where={"lifecycle": "implemented"})
    assert {h.chunk_id for h in hits} == {"c-impl"}

    hits = store.search(_vec(0.1), k=5, where={"lifecycle": "superseded"})
    assert hits == []

    hits = store.search(_vec(0.1), k=5)
    assert {h.chunk_id for h in hits} == {"c-impl", "c-prop", "c-asis"}


def test_search_with_where_in_operator(store: ChromaStore) -> None:
    chunks = [
        Chunk(id="c-impl", doc_id="d-1", text="t1", metadata={"lifecycle": "implemented"}),
        Chunk(id="c-prop", doc_id="d-2", text="t2", metadata={"lifecycle": "proposed"}),
        Chunk(id="c-asis", doc_id="d-3", text="t3", metadata={"lifecycle": "as-is"}),
    ]
    store.upsert(chunks, [_vec(0.1), _vec(0.2), _vec(0.3)])

    hits = store.search(
        _vec(0.1), k=5,
        where={"lifecycle": {"$in": ["implemented", "proposed"]}},
    )
    assert {h.chunk_id for h in hits} == {"c-impl", "c-prop"}
