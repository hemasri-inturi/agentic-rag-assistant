"""FAISS-backed retrieval over document chunks.

Chunks are embedded with :class:`HashEmbedder` (offline, deterministic)
and stored in a FAISS ``IndexFlatIP`` index over unit-norm vectors,
which is equivalent to cosine similarity.

Persistence: when a ``persist_dir`` is given, every mutation saves the
index (``faiss.write_index``) and the chunk list (JSON) to that
directory, and :meth:`FaissStore.load` restores them on startup, so
ingested documents survive restarts. When no persisted index exists
(or it is corrupt / built for a different embedding dimension), the
store simply starts empty and the caller falls back to loading the
bundled documents — in-memory behaviour is preserved as the fallback,
never a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from .embeddings import HashEmbedder
from .logging_config import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


class FaissStore:
    """FAISS vector store for document chunks, with disk persistence."""

    def __init__(self, dim: int = 384, persist_dir: str | Path | None = None) -> None:
        self.embedder = HashEmbedder(dim=dim)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Document] = []
        self.persist_dir = Path(persist_dir) if persist_dir else None

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
        self.save()
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

    def save(self) -> None:
        """Persist the index + chunks to ``persist_dir`` (no-op if unset).

        A save failure is logged, not raised: indexing must not take
        the API down, and the in-memory index remains usable.
        """
        if self.persist_dir is None:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(
                self.index, str(self.persist_dir / INDEX_FILENAME)
            )
            payload = [
                {"page_content": c.page_content, "metadata": dict(c.metadata)}
                for c in self.chunks
            ]
            (self.persist_dir / CHUNKS_FILENAME).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            logger.info(
                "persisted %d chunks to %s", len(self.chunks), self.persist_dir
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("failed to persist index to %s: %s", self.persist_dir, exc)

    def load(self) -> bool:
        """Restore a persisted index + chunks. Returns True on success.

        Returns False when there is nothing to load, the files are
        corrupt, or the stored index was built for a different
        embedding dimension. In all False cases the store is left
        empty and the caller falls back to its normal startup path.
        """
        if self.persist_dir is None:
            return False
        index_path = self.persist_dir / INDEX_FILENAME
        chunks_path = self.persist_dir / CHUNKS_FILENAME
        if not index_path.is_file() or not chunks_path.is_file():
            return False
        try:
            index = faiss.read_index(str(index_path))
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [
                Document(
                    page_content=item["page_content"],
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload
            ]
        except Exception as exc:  # noqa: BLE001 - corrupt store = fresh start
            logger.warning(
                "persisted index at %s unreadable, starting empty: %s",
                self.persist_dir,
                exc,
            )
            return False
        if index.d != self.embedder.dim or index.ntotal != len(chunks):
            logger.warning(
                "persisted index at %s does not match current settings "
                "(dim %s vs %s, vectors %s vs %s chunks); starting empty",
                self.persist_dir,
                index.d,
                self.embedder.dim,
                index.ntotal,
                len(chunks),
            )
            return False
        self.index = index
        self.chunks = chunks
        logger.info(
            "loaded persisted index: %d chunks from %s",
            len(chunks),
            self.persist_dir,
        )
        return True
