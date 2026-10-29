"""Deterministic, offline embeddings.

This project deliberately uses a hashed bag-of-words embedder instead
of a hosted embedding API, so retrieval works fully offline and the
only network call in the whole system is the optional LLM chat call.
Tokens are hashed into a fixed-dimension vector and L2-normalised, so
cosine similarity via FAISS inner product is meaningful for keyword
overlap. This is a documented simplification, not a claim of
state-of-the-art retrieval quality.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-norm float32 vector for ``text``."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])
