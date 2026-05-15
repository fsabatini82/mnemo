"""Typed runtime configuration for Mnemo.

Two knowledge collections coexist in the same RAG store: `specs` and
`bug_memory`. Each is namespaced under a `(project, environment)` pair
so a single Mnemo install can serve multiple projects and lifecycle
stages (dev, col, pre, prd) without cross-contamination.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

StoreName = Literal["chroma", "lance"]
PipelineName = Literal["default", "llamaindex"]
EnvironmentName = Literal["dev", "col", "pre", "prd"]

_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class Settings(BaseSettings):
    # Backend selection
    store: StoreName = "chroma"
    pipeline: PipelineName = "default"

    # Embedding model
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Persistence
    persist_dir: Path = Path("./data")

    # --- Multi-tenant namespace ---------------------------------------------
    # `project` is the human-facing slug; the registry maps it to a stable
    # 3-digit ID used in the underlying collection names. `environment`
    # is the lifecycle stage (dev/col/pre/prd) — collections are scoped
    # by (project_id, environment) so the same project can keep separate
    # corpora per environment.
    project: str = "demo-project"
    environment: EnvironmentName = "dev"

    # Collection axes — these are the suffixes of the final collection names.
    specs_collection: str = "specs"
    bugs_collection: str = "bug_memory"

    # External source paths (the ingestion CLIs read from these)
    specs_source_dir: Path = Path("./assets/specs-source")
    bugs_source_dir: Path = Path("./assets/bugs-source")

    # Chunking
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # Retrieval
    top_k: int = Field(default=5, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="MNEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ validators

    @field_validator("project")
    @classmethod
    def _validate_project_slug(cls, v: str) -> str:
        if not _SLUG_RE.fullmatch(v):
            raise ValueError(
                f"Invalid project slug {v!r}: must be lowercase, start with a "
                "letter, max 32 chars, and contain only [a-z0-9-]."
            )
        return v

    @field_validator("specs_collection", "bugs_collection")
    @classmethod
    def _validate_axis(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z_]+", v):
            raise ValueError(
                f"Collection axis {v!r} must be lowercase letters/underscore only."
            )
        return v


def load_settings() -> Settings:
    return Settings()
