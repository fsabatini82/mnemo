"""Factory: turns `Settings` into a `MnemoSystem` with two pipelines.

Mnemo holds two parallel knowledge collections (`specs`, `bug_memory`).
The factory builds one shared embedder and one `RagPipeline` per
collection, both pointing to the same physical store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mnemo.config import Settings
from mnemo.core.protocols import Embedder, RagPipeline, VectorStore


@dataclass(slots=True, frozen=True)
class MnemoSystem:
    """Wires the consumer side: two pipelines + the source settings."""

    specs: RagPipeline
    bugs: RagPipeline
    settings: Settings


def build_system(settings: Settings) -> MnemoSystem:
    if settings.pipeline == "llamaindex":
        try:
            from mnemo.pipelines.llamaindex import LlamaIndexPipeline
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex pipeline requested but extras not installed. "
                'Run: pip install -e ".[llamaindex]"'
            ) from exc
        specs = LlamaIndexPipeline(
            embed_model=settings.embed_model,
            persist_dir=settings.persist_dir,
            collection=settings.specs_collection,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        bugs = LlamaIndexPipeline(
            embed_model=settings.embed_model,
            persist_dir=settings.persist_dir,
            collection=settings.bugs_collection,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return MnemoSystem(specs=specs, bugs=bugs, settings=settings)

    from mnemo.embedders.fastembed_embedder import FastEmbedEmbedder
    from mnemo.pipelines.default import DefaultPipeline

    embedder = FastEmbedEmbedder(model_name=settings.embed_model)
    specs_store = _build_store(settings, dimension=embedder.dimension, collection=settings.specs_collection)
    bugs_store = _build_store(settings, dimension=embedder.dimension, collection=settings.bugs_collection)

    specs = DefaultPipeline(
        embedder=embedder,
        store=specs_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    bugs = DefaultPipeline(
        embedder=embedder,
        store=bugs_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return MnemoSystem(specs=specs, bugs=bugs, settings=settings)


# Backwards-friendly alias for code that still wants a single pipeline.
def build_pipeline(settings: Settings) -> RagPipeline:
    return build_system(settings).specs


def _build_store(settings: Settings, *, dimension: int, collection: str) -> VectorStore:
    if settings.store == "chroma":
        from mnemo.stores.chroma_store import ChromaStore

        return ChromaStore(
            persist_dir=settings.persist_dir,
            collection=collection,
        )

    if settings.store == "lance":
        try:
            from mnemo.stores.lance_store import LanceStore
        except ImportError as exc:
            raise RuntimeError(
                "LanceDB store requested but extras not installed. "
                'Run: pip install -e ".[lance]"'
            ) from exc
        return LanceStore(
            persist_dir=settings.persist_dir,
            collection=collection,
            dimension=dimension,
        )

    raise ValueError(f"Unknown store: {settings.store}")
