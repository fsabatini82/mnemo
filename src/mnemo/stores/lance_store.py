"""LanceDB vector store with optional hybrid search (vector + FTS).

Requires the `[lance]` extras. Satisfies both `VectorStore` and
`SupportsHybridSearch` Protocols — the pipeline will pick the hybrid path
automatically when the store advertises it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.rerankers import RRFReranker

from mnemo.core.models import Chunk, Hit
from mnemo.core.protocols import matches_where as _matches_where


class LanceStore:
    _TABLE_SUFFIX = "chunks"

    def __init__(self, persist_dir: Path, collection: str, dimension: int) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(persist_dir))
        self._dimension = dimension
        self._table_name = f"{collection}_{self._TABLE_SUFFIX}"
        self._table = self._ensure_table()
        self._reranker = RRFReranker()

    def _ensure_table(self) -> Any:
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                pa.field("metadata", pa.string()),
            ]
        )
        return self._db.create_table(self._table_name, schema=schema, mode="create")

    def upsert(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return
        rows = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "vector": list(e),
                "metadata": json.dumps(c.metadata),
            }
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        # FTS index enables the keyword side of hybrid search.
        # `replace=False` is a no-op when the index already exists.
        try:
            self._table.create_fts_index("text", replace=False)
        except Exception:
            pass

    def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        # Lance stores metadata as a JSON-encoded text column, so we
        # can't push arbitrary `key=value` filters into the SQL WHERE
        # without column-promoting each filterable field. We over-fetch
        # and filter client-side. For lab-scale corpora this is fine;
        # for very large stores, promote the relevant metadata field
        # to a top-level Lance column.
        fetch_k = k if where is None else max(k * 4, 20)
        rows = self._table.search(list(embedding)).limit(fetch_k).to_list()
        hits = self._to_hits(rows, score_key="_distance", invert_distance=True)
        if where:
            hits = [h for h in hits if _matches_where(h.metadata, where)]
        return hits[:k]

    def hybrid_search(
        self,
        embedding: Sequence[float],
        query_text: str,
        *,
        k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[Hit]:
        fetch_k = k if where is None else max(k * 4, 20)
        rows = (
            self._table.search(query_type="hybrid")
            .vector(list(embedding))
            .text(query_text)
            .rerank(reranker=self._reranker)
            .limit(fetch_k)
            .to_list()
        )
        hits = self._to_hits(rows, score_key="_relevance_score", invert_distance=False)
        if where:
            hits = [h for h in hits if _matches_where(h.metadata, where)]
        return hits[:k]

    @staticmethod
    def _to_hits(
        rows: list[dict[str, Any]],
        *,
        score_key: str,
        invert_distance: bool,
    ) -> list[Hit]:
        hits: list[Hit] = []
        for r in rows:
            raw = float(r.get(score_key, 0.0))
            score = 1.0 - raw if invert_distance else raw
            hits.append(
                Hit(
                    chunk_id=r["id"],
                    doc_id=r["doc_id"],
                    text=r["text"],
                    score=score,
                    metadata=json.loads(r.get("metadata") or "{}"),
                )
            )
        return hits


