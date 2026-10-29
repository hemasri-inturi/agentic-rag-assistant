"""Tests for the FAISS retriever."""

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag_assistant.retriever import FaissStore


def _store_with_docs() -> FaissStore:
    store = FaissStore(dim=384)
    store.add_documents(
        [
            Document(
                page_content=(
                    "Retrieval-Augmented Generation combines retrieval with "
                    "generation to answer questions from documents."
                ),
                metadata={"title": "RAG doc"},
            ),
            Document(
                page_content=(
                    "A virtual environment isolates Python packages per "
                    "project so dependencies do not conflict."
                ),
                metadata={"title": "Python doc"},
            ),
        ]
    )
    return store


def test_search_returns_relevant_chunk_first():
    """A RAG question retrieves the RAG chunk, not the Python chunk."""
    store = _store_with_docs()
    results = store.search("What is Retrieval-Augmented Generation?", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["title"] == "RAG doc"


def test_search_empty_store_returns_empty():
    """Searching an empty store returns [] instead of erroring."""
    assert FaissStore(dim=384).search("anything") == []


def test_search_empty_query_returns_empty():
    """A blank query retrieves nothing."""
    assert _store_with_docs().search("   ") == []


def test_add_documents_counts():
    """add_documents reports how many chunks were added."""
    store = FaissStore(dim=384)
    assert store.add_documents([]) == 0
    assert store.add_documents(
        [Document(page_content="hello world", metadata={})]
    ) == 1
    assert len(store) == 1
