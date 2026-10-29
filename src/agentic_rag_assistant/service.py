"""Service layer: ties the store, the graph and ingestion together.

The API layer talks only to this class, never directly to FAISS or
LangGraph, which keeps the HTTP code thin and the agent logic testable
without a web server.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .config import Settings, load_settings
from .documents import load_markdown_files, split_documents
from .graph import build_graph
from .logging_config import configure_logging, get_logger
from .retriever import FaissStore

configure_logging()
logger = get_logger(__name__)


class AssistantService:
    """Owns the vector store and the compiled agent graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.store = FaissStore(
            dim=self.settings.embedding_dim,
            persist_dir=self.settings.index_dir,
        )
        self.graph = build_graph(self.store, self.settings)
        self._sample_docs_loaded = False

    def load_sample_docs(self) -> int:
        """Prepare the index (once per service lifetime).

        If a persisted index exists it is restored (including any
        previously ingested documents) and the bundled docs are NOT
        re-added, so a restart never duplicates chunks. Otherwise the
        bundled synthetic docs are indexed and persisted for next time.
        Returns the number of newly indexed chunks (0 on restore).
        """
        if self._sample_docs_loaded:
            return 0
        if self.store.load():
            self._sample_docs_loaded = True
            return 0
        docs = load_markdown_files(self.settings.data_dir)
        chunks = split_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.store.add_documents(chunks)
        self._sample_docs_loaded = True
        return added

    def ingest_texts(
        self, texts: list[tuple[str, str, str]]
    ) -> tuple[int, int]:
        """Ingest ``(doc_id, title, text)`` triples.

        Returns ``(chunks_added, total_chunks_indexed)``.
        """
        docs = [
            Document(
                page_content=text,
                metadata={"source": doc_id, "title": title},
            )
            for doc_id, title, text in texts
        ]
        chunks = split_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.store.add_documents(chunks)
        return added, len(self.store)

    def ask(self, question: str) -> dict:
        """Run the agent graph for one question and return a plain dict."""
        if not question.strip():
            raise ValueError("question must not be empty")
        # Lazy load: guarantees the bundled docs are present even when
        # the app is used without running the startup lifespan (e.g.
        # TestClient without a context manager).
        if not self._sample_docs_loaded:
            self.load_sample_docs()
        result = self.graph.invoke({"question": question})
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "route": result.get("route", "factual"),
            "critic_passed": result.get("critic_passed", False),
            "llm_used": result.get("llm_used", False),
        }

    @property
    def docs_indexed(self) -> int:
        """Number of chunks currently in the index."""
        return len(self.store)
