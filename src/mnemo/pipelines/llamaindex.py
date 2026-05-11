"""LlamaIndex-backed pipeline (optional, requires `[llamaindex]` extras).

Wraps a `VectorStoreIndex` behind the same `RagPipeline` Protocol the rest
of the system consumes — shown as the "what you'd reach for in production
when you outgrow the hand-rolled pipeline" alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LIDocument
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from mnemo.core.models import Document, Hit, QueryResult


class LlamaIndexPipeline:
    def __init__(
        self,
        *,
        embed_model: str,
        persist_dir: Path,
        collection: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        chroma_collection = client.get_or_create_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._embed_model = FastEmbedEmbedding(model_name=embed_model)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._embed_model,
            storage_context=storage_context,
        )
        self._parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest(self, documents: Sequence[Document]) -> None:
        li_docs = [
            LIDocument(text=d.text, doc_id=d.id, metadata=d.metadata) for d in documents
        ]
        nodes = self._parser.get_nodes_from_documents(li_docs)
        self._index.insert_nodes(nodes)

    def query(self, question: str, *, k: int = 5) -> QueryResult:
        retriever = self._index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(question)
        hits = [
            Hit(
                chunk_id=r.node.node_id,
                doc_id=str(r.node.metadata.get("doc_id", r.node.ref_doc_id or "")),
                text=r.node.get_content(),
                score=float(r.score or 0.0),
                metadata=dict(r.node.metadata),
            )
            for r in results
        ]
        return QueryResult(question=question, hits=hits)
