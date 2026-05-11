"""FastEmbed-backed embedder.

FastEmbed ships ONNX-quantized models — no PyTorch, fast cold start, fully
local. Satisfies `Embedder` Protocol structurally.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    @cached_property
    def dimension(self) -> int:
        probe = next(iter(self._model.embed(["probe"])))
        return len(probe)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]
