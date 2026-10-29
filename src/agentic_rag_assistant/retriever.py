"""FAISS-backed retrieval over document chunks.

Chunks are embedded with :class:`HashEmbedder` (offline, deterministic)
and stored in a FAISS ``IndexFlatIP`` index over unit-norm vectors,
which is equivalent to cosine similarity. The store is in-memory by
design for this portfolio project; persistence is a documented
limitation, not a hidden gap.
"""

from __future__ import annotations

import faiss
import numpy as np
from langchain_core.documents import Document

from .embeddings import HashEmbedder
from .logging_config import get_logger

logger = get_logger(__name__)


class FaissStore:
    """In-memory FAISS vector store for document chunks."""

    def __init__(self, dim: int = 384) -> None:
        self.embedder = HashEmbedder(dim=dim)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Document] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add_documents(self, chunks: list[Document]) -> int:
        """Add chunks to the index. Returns the number added."""
        if not chunks:
            return 0
        vectors = self.embedder.embed_many([c.page_content for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(
            "indexed %d chunks (%d total)", len(chunks), len(self.chunks)
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[Document]:
        """Return the ``top_k`` most similar chunks for ``query``."""
        if len(self.chunks) == 0 or not query.strip():
            return []
        query_vec = self.embedder.embed(query).reshape(1, -1)
        k = min(top_k, len(self.chunks))
        _scores, indices = self.index.search(query_vec, k)
        results: list[Document] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            results.append(self.chunks[int(idx)])
        return results
