"""Chroma persistent vector store (default backend).

Satisfies `VectorStore` Protocol. Cosine similarity, telemetry disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[list(e) for e in embeddings],
            metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in chunks],
        )

    def search(self, embedding: Sequence[float], *, k: int = 5) -> list[Hit]:
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
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
