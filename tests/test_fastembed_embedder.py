"""Tests for the FastEmbed-backed embedder (with the underlying model stubbed)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest


class _FakeTextEmbedding:
    """Drop-in stand-in for `fastembed.TextEmbedding` — no model download.

    Mimics the real return type (numpy arrays) so the wrapper's `.tolist()`
    works.
    """

    DIM = 6

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def embed(self, texts: list[str]) -> Iterator[Any]:
        for _ in texts:
            yield np.array([0.1] * self.DIM, dtype=np.float32)


@pytest.fixture
def patched_text_embedding(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    monkeypatch.setattr(
        "mnemo.embedders.fastembed_embedder.TextEmbedding",
        _FakeTextEmbedding,
    )
    return _FakeTextEmbedding


def test_embedder_dimension_probes_underlying_model(
    patched_text_embedding: type[_FakeTextEmbedding],
) -> None:
    from mnemo.embedders.fastembed_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model_name="fake")
    assert embedder.dimension == _FakeTextEmbedding.DIM


def test_embedder_embed_returns_python_lists(
    patched_text_embedding: type[_FakeTextEmbedding],
) -> None:
    from mnemo.embedders.fastembed_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model_name="fake")
    out = embedder.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(isinstance(vec, list) for vec in out)
    assert all(len(vec) == _FakeTextEmbedding.DIM for vec in out)


def test_embedder_dimension_is_cached(
    patched_text_embedding: type[_FakeTextEmbedding],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnemo.embedders.fastembed_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model_name="fake")
    first = embedder.dimension
    second = embedder.dimension
    assert first == second  # cached_property returns the same value
