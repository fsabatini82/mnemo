"""Typed runtime configuration for Mnemo.

Two knowledge collections coexist in the same RAG store: `specs` and
`bug_memory`. Source paths are configurable so the audience can point
them at the asset folders extracted from the lab repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

StoreName = Literal["chroma", "lance"]
PipelineName = Literal["default", "llamaindex"]


class Settings(BaseSettings):
    # Backend selection
    store: StoreName = "chroma"
    pipeline: PipelineName = "default"

    # Embedding model
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Persistence
    persist_dir: Path = Path("./data")

    # Collection names — one per knowledge axis
    specs_collection: str = "specs"
    bugs_collection: str = "bug_memory"

    # External source paths (the agentic CLIs read from these)
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


def load_settings() -> Settings:
    return Settings()
