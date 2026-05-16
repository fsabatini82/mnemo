"""Factory: turns `Settings` into a `MnemoSystem` with two pipelines.

Mnemo holds two parallel knowledge collections (`specs`, `bug_memory`),
namespaced by `(project_id, environment)`. The factory builds one
shared embedder and one `RagPipeline` per collection, both pointing
to the same physical store.

Project ID resolution: the factory consults the project registry at
`<persist_dir>/projects.json`. If the configured project is not yet
registered, `build_system` does NOT auto-register — read-only by
design. Writers (`mnemo-ingest`, `mnemo-admin`) own that side effect.
The MCP server, on the other hand, always reads — so if the project
is missing we fall back to a deterministic "ephemeral" id derived
from the slug, with a warning logged so the user notices.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from mnemo.config import Settings
from mnemo.core.protocols import RagPipeline, VectorStore
from mnemo.registry import (
    ProjectRegistry,
    collection_name,
    open_registry,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class MnemoSystem:
    """Wires the consumer side: three pipelines + the source settings."""

    specs: RagPipeline
    bugs: RagPipeline
    devops: RagPipeline
    settings: Settings
    project_id: str
    environment: str

    @property
    def effective_prefix(self) -> str:
        return f"{self.project_id}_{self.environment}"


def build_system(
    settings: Settings,
    *,
    registry: ProjectRegistry | None = None,
    auto_register: bool = False,
) -> MnemoSystem:
    """Build the consumer system for the configured `(project, environment)`.

    Args:
        settings: Loaded `Settings`.
        registry: Optional pre-loaded registry. If None, opens one at
            `settings.persist_dir`.
        auto_register: If True, register the project (and environment)
            in the registry on the fly. The CLI passes `auto_register=True`
            on ingestion; the MCP server keeps it False (read-only).
    """
    registry = registry or open_registry(settings.persist_dir)
    project_id = _resolve_project_id(settings, registry, auto_register=auto_register)
    env = settings.environment

    specs_name = collection_name(project_id, env, settings.specs_collection)
    bugs_name = collection_name(project_id, env, settings.bugs_collection)
    devops_name = collection_name(project_id, env, settings.devops_collection)

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
            collection=specs_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        bugs = LlamaIndexPipeline(
            embed_model=settings.embed_model,
            persist_dir=settings.persist_dir,
            collection=bugs_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        devops = LlamaIndexPipeline(
            embed_model=settings.embed_model,
            persist_dir=settings.persist_dir,
            collection=devops_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return MnemoSystem(
            specs=specs, bugs=bugs, devops=devops, settings=settings,
            project_id=project_id, environment=env,
        )

    from mnemo.embedders.fastembed_embedder import FastEmbedEmbedder
    from mnemo.pipelines.default import DefaultPipeline

    embedder = FastEmbedEmbedder(model_name=settings.embed_model)
    specs_store = _build_store(settings, dimension=embedder.dimension, collection=specs_name)
    bugs_store = _build_store(settings, dimension=embedder.dimension, collection=bugs_name)
    devops_store = _build_store(settings, dimension=embedder.dimension, collection=devops_name)

    specs = DefaultPipeline(
        embedder=embedder, store=specs_store,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    bugs = DefaultPipeline(
        embedder=embedder, store=bugs_store,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    devops = DefaultPipeline(
        embedder=embedder, store=devops_store,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    return MnemoSystem(
        specs=specs, bugs=bugs, devops=devops, settings=settings,
        project_id=project_id, environment=env,
    )


# Backwards-friendly alias for code that still wants a single pipeline.
def build_pipeline(settings: Settings) -> RagPipeline:
    return build_system(settings).specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_project_id(
    settings: Settings,
    registry: ProjectRegistry,
    *,
    auto_register: bool,
) -> str:
    record = registry.get(settings.project)
    if record is not None:
        return record.id

    if auto_register:
        record = registry.ensure(settings.project)
        registry.add_environment(settings.project, settings.environment)
        registry.save()
        logger.info(
            "Registered project %r as id=%s (env=%s)",
            settings.project, record.id, settings.environment,
        )
        return record.id

    # Read-only path (mcp-server): emit a stable ephemeral id derived
    # from the slug, log a warning. The next ingestion will overwrite
    # this with a real registry entry.
    ephemeral = _ephemeral_id(settings.project)
    logger.warning(
        "Project %r not registered yet. Using ephemeral id=%s for read-only "
        "access. Run `mnemo-ingest --project %s --env %s` to register.",
        settings.project, ephemeral, settings.project, settings.environment,
    )
    return ephemeral


def _ephemeral_id(slug: str) -> str:
    """Deterministic 3-digit hash of the slug for read-only fallback."""
    h = hashlib.sha256(slug.encode("utf-8")).digest()
    n = int.from_bytes(h[:2], "big") % 1000
    return f"{n:03d}"


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
