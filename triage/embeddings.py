"""Text embedders for the change-record index.

`fastembed` runs a small local model, so retrieval needs no API key. `hash` is a
deterministic bag-of-words embedder for tests and CI, where downloading a model is
not worth it. Pick one with EMBEDDING_PROVIDER (default: fastembed).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    dim = 384

    def __init__(self, model: str = FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        cache = os.environ.get("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model, cache_dir=cache)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashEmbedder:
    """Hashes word tokens into a fixed-size, L2-normalised vector."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(provider: str | None = None) -> Embedder:
    provider = provider or os.environ.get("EMBEDDING_PROVIDER", "fastembed")
    if provider == "fastembed":
        return FastEmbedder()
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}; use 'fastembed' or 'hash'")
