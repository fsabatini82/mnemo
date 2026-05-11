"""Token-aware chunking via `langchain-text-splitters` (production-ready).

We use `RecursiveCharacterTextSplitter` with a tiktoken-backed length
function so chunk_size is measured in tokens, not characters. Separators
are ordered from coarse (markdown headings) to fine (whitespace) — the
splitter tries them in order and falls back to the next one when a chunk
is still too large.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mnemo.core.models import Chunk, Document


def build_splitter(*, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in documents:
        for idx, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(
                    id=f"{doc.id}::chunk-{idx:04d}",
                    doc_id=doc.id,
                    text=piece,
                    metadata={**doc.metadata, "chunk_index": idx},
                )
            )
    return chunks
