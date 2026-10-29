"""Tests for FAISS disk persistence (ingest -> reload -> ask)."""

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag_assistant.config import Settings
from agentic_rag_assistant.retriever import FaissStore
from agentic_rag_assistant.service import AssistantService


def test_persistence_round_trip(tmp_path):
    """Chunks saved by one store are searchable by a fresh store."""
    index_dir = tmp_path / "index"
    store = FaissStore(dim=384, persist_dir=index_dir)
    store.add_documents(
        [
            Document(
                page_content=(
                    "The quokkaflux protocol synchronises widgets across "
                    "devices using a shared ledger."
                ),
                metadata={"title": "Quokkaflux doc"},
            )
        ]
    )
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "chunks.json").is_file()

    # Fresh store = restart equivalent: load, then search.
    reloaded = FaissStore(dim=384, persist_dir=index_dir)
    assert reloaded.load() is True
    assert len(reloaded) == len(store)
    results = reloaded.search("quokkaflux protocol synchronises widgets")
    assert results and results[0].metadata["title"] == "Quokkaflux doc"


def test_load_without_persisted_files_returns_false(tmp_path):
    """A store with no persisted files starts empty, no crash."""
    store = FaissStore(dim=384, persist_dir=tmp_path / "missing")
    assert store.load() is False
    assert len(store) == 0


def test_load_with_wrong_dimension_returns_false(tmp_path):
    """An index built for another dimension is ignored, not loaded."""
    index_dir = tmp_path / "index"
    FaissStore(dim=384, persist_dir=index_dir).add_documents(
        [Document(page_content="hello world", metadata={})]
    )
    mismatched = FaissStore(dim=128, persist_dir=index_dir)
    assert mismatched.load() is False
    assert len(mismatched) == 0


def test_service_restart_finds_ingested_doc(tmp_path):
    """Ingest -> new service instance -> ask finds the ingested doc."""
    empty_docs = tmp_path / "docs"
    empty_docs.mkdir()
    index_dir = tmp_path / "index"
    settings = Settings(
        data_dir=str(empty_docs),
        index_dir=str(index_dir),
    )
    first = AssistantService(settings=settings)
    first.load_sample_docs()
    first.ingest_texts(
        [
            (
                "restart-doc",
                "Zebraquantum doc",
                "The zebraquantum protocol synchronises widgets across "
                "devices using a shared ledger.",
            )
        ]
    )

    # Restart equivalent: a brand-new service on the same index dir.
    restarted = AssistantService(settings=settings)
    restarted.load_sample_docs()
    result = restarted.ask("zebraquantum protocol synchronises widgets")
    assert "zebraquantum" in result["answer"].lower()
    assert "Zebraquantum doc" in result["sources"]
