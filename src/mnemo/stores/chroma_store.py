"""Chroma persistent vector store (default backend).

Satisfies `VectorStore` Protocol. Cosine similarity, telemetry disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from mnemo.core.models import Chunk, Hit


class ChromaStore:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # Chroma's rust core caps a single upsert at 5461 items; batch below that.
    _UPSERT_BATCH = 5000

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return
        for start in range(0, len(chunks), self._UPSERT_BATCH):
            end = start + self._UPSERT_BATCH
            batch = chunks[start:end]
            emb = embeddings[start:end]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=[list(e) for e in emb],
                metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in batch],
            )

    def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            # Chroma accepts equality filters as `where={"key": "value"}`
            # and supports operators like `{"$in": [...]}` natively.
            query_kwargs["where"] = dict(where)
        result = self._collection.query(**query_kwargs)
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            Hit(
                chunk_id=cid,
                doc_id=str(meta.get("doc_id", "")),
                text=doc,
                score=1.0 - float(dist),
                metadata={k_: v for k_, v in meta.items() if k_ != "doc_id"},
            )
            for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True)
        ]
