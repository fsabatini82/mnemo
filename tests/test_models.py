"""Tests for the core data carriers."""

from __future__ import annotations

import pytest

from mnemo.core.models import Chunk, Document, Hit, QueryResult


def test_document_defaults_empty_metadata() -> None:
    doc = Document(id="x", text="hello")
    assert doc.metadata == {}


def test_document_carries_metadata() -> None:
    doc = Document(id="x", text="hello", metadata={"k": 1})
    assert doc.metadata == {"k": 1}


def test_document_is_frozen() -> None:
    doc = Document(id="x", text="hello")
    with pytest.raises(Exception):  # FrozenInstanceError, attribute, etc.
        doc.id = "y"  # type: ignore[misc]


def test_chunk_basic() -> None:
    chunk = Chunk(id="c-1", doc_id="d-1", text="body", metadata={"i": 0})
    assert chunk.doc_id == "d-1"
    assert chunk.metadata == {"i": 0}


def test_hit_carries_score() -> None:
    h = Hit(chunk_id="c", doc_id="d", text="t", score=0.42, metadata={})
    assert h.score == pytest.approx(0.42)


def test_query_result_contains_hits() -> None:
    hits = [Hit(chunk_id=f"c{i}", doc_id="d", text=f"t{i}", score=1.0 - i * 0.1) for i in range(3)]
    qr = QueryResult(question="q?", hits=hits)
    assert qr.question == "q?"
    assert len(qr.hits) == 3
    assert qr.hits[0].chunk_id == "c0"


def test_metadata_default_factories_are_independent() -> None:
    a = Document(id="a", text="x")
    b = Document(id="b", text="x")
    # Frozen dataclasses can't mutate `metadata` attribute directly, but
    # the dict they wrap is mutable. Each instance gets its own dict.
    a.metadata["k"] = 1
    assert b.metadata == {}
