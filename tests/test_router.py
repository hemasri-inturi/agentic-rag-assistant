"""Tests for the router and the critic grounding check."""

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag_assistant.config import Settings
from agentic_rag_assistant.graph import build_graph, route_question
from agentic_rag_assistant.retriever import FaissStore


def test_router_greeting_is_general():
    """Greetings skip retrieval."""
    assert route_question("Hello") == "general"
    assert route_question("Hi there") == "general"


def test_router_meta_question_is_general():
    """Meta questions about the assistant itself skip retrieval."""
    assert route_question("What can you do?") == "general"


def test_router_factual_question():
    """Ordinary topic questions go to retrieval."""
    assert route_question("What is RAG?") == "factual"
    assert route_question("How do I create a virtual environment?") == "factual"


def test_graph_factual_answer_is_grounded():
    """A factual question returns a grounded answer with sources."""
    store = FaissStore(dim=384)
    store.add_documents(
        [
            Document(
                page_content=(
                    "Retrieval-Augmented Generation combines retrieval with "
                    "generation to answer questions from documents."
                ),
                metadata={"title": "RAG doc"},
            )
        ]
    )
    graph = build_graph(store, Settings())
    result = graph.invoke({"question": "What is Retrieval-Augmented Generation?"})
    assert result["route"] == "factual"
    assert result["critic_passed"] is True
    assert "retrieval" in result["answer"].lower()
    assert result["sources"] == ["RAG doc"]
    assert result["llm_used"] is False  # offline fallback in tests


def test_graph_general_answer_skips_retrieval():
    """A greeting gets a general answer and no sources."""
    store = FaissStore(dim=384)
    graph = build_graph(store, Settings())
    result = graph.invoke({"question": "Hello"})
    assert result["route"] == "general"
    assert result["sources"] == []
    assert result["critic_passed"] is True
