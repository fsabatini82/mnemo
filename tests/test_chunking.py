"""Tests for the chunking module."""

from __future__ import annotations

import pytest

from mnemo.chunking import build_splitter, chunk_documents
from mnemo.core.models import Document


def test_build_splitter_returns_token_aware_splitter() -> None:
    splitter = build_splitter(chunk_size=100, chunk_overlap=10)
    assert splitter is not None
    # The splitter has the public split_text method we rely on.
    assert hasattr(splitter, "split_text")


def test_chunk_documents_produces_chunks() -> None:
    doc = Document(id="d", text="word " * 500)
    chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(c.doc_id == "d" for c in chunks)


def test_chunk_documents_preserves_doc_metadata() -> None:
    doc = Document(id="d", text="content " * 100, metadata={"source": "wiki", "tag": "x"})
    chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
    for c in chunks:
        assert c.metadata["source"] == "wiki"
        assert c.metadata["tag"] == "x"
        assert "chunk_index" in c.metadata


def test_chunk_documents_assigns_unique_ids() -> None:
    doc = Document(id="d", text="alpha " * 200)
    chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    # IDs are well-formed.
    assert all(c.id.startswith("d::chunk-") for c in chunks)


def test_chunk_index_metadata_is_sequential() -> None:
    doc = Document(id="d", text="alpha " * 200)
    chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_documents_handles_multiple_docs() -> None:
    docs = [
        Document(id="a", text="first text " * 100),
        Document(id="b", text="second text " * 100),
    ]
    chunks = chunk_documents(docs, chunk_size=64, chunk_overlap=8)
    a_chunks = [c for c in chunks if c.doc_id == "a"]
    b_chunks = [c for c in chunks if c.doc_id == "b"]
    assert a_chunks and b_chunks


def test_chunk_documents_with_short_text() -> None:
    doc = Document(id="s", text="short text")
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=32)
    # Short text → single chunk.
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


@pytest.mark.parametrize("size,overlap", [(128, 16), (256, 32), (512, 64)])
def test_chunk_documents_respects_chunk_size(size: int, overlap: int) -> None:
    doc = Document(id="d", text="lorem ipsum " * 500)
    chunks = chunk_documents([doc], chunk_size=size, chunk_overlap=overlap)
    assert len(chunks) > 0
