"""The multi-agent flow, built with LangGraph.

Pipeline: ``router -> retrieval -> answer -> critic``.

* **Router** — a small rule-based classifier. It routes greetings and
  meta questions ("what can you do?") to a general answer and everything
  else to retrieval. Rule-based routing is a deliberate choice for a
  portfolio project: it is testable offline and its decisions are
  explainable, unlike an LLM call that would also cost a request.
* **Retrieval** — FAISS search over the indexed chunks.
* **Answer** — LLM answer when configured, extractive fallback otherwise.
* **Critic** — a self-check: the answer must share meaningful content
  words with the retrieved chunks. If it does not, the answer is
  replaced with an honest "not found in the documents" reply and the
  sources are kept, so an ungrounded answer never reaches the user.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .embeddings import tokenize
from .llm import extractive_answer, generate_answer
from .logging_config import get_logger
from .retriever import FaissStore

logger = get_logger(__name__)

_GREETING_RE = re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b")
_META_RE = re.compile(
    r"\b(what can you do|who are you|how do you work|help)\b"
)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "which", "who", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "do", "does", "can", "you", "i",
}


class AgentState(TypedDict, total=False):
    """State carried through the agent graph."""

    question: str
    route: str
    chunks: list[Document]
    answer: str
    sources: list[str]
    critic_passed: bool
    llm_used: bool


def route_question(question: str) -> str:
    """Classify a question as ``"general"`` or ``"factual"``.

    Pure function, so it is unit-testable without any model or index.
    """
    text = question.strip().lower()
    if _GREETING_RE.match(text) or _META_RE.search(text):
        return "general"
    return "factual"


def _content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def build_graph(store: FaissStore, settings: Settings):
    """Compile the router -> retrieval -> answer -> critic graph."""
    top_k = settings.top_k

    def router_node(state: AgentState) -> AgentState:
        route = route_question(state["question"])
        logger.info("router: %r -> %s", state["question"], route)
        return {"route": route}

    def retrieval_node(state: AgentState) -> AgentState:
        if state.get("route") != "factual":
            return {"chunks": [], "sources": []}
        chunks = store.search(state["question"], top_k=top_k)
        sources = [c.metadata.get("title", "unknown") for c in chunks]
        return {"chunks": chunks, "sources": sources}

    def answer_node(state: AgentState) -> AgentState:
        if state.get("route") != "factual":
            return {
                "answer": (
                    "Hello! I answer questions from the documents indexed "
                    "in this assistant. Ask me something covered by those "
                    "documents, or ingest your own documents first."
                ),
                "llm_used": False,
            }
        answer, llm_used = generate_answer(
            settings, state["question"], state.get("chunks", [])
        )
        return {"answer": answer, "llm_used": llm_used}

    def critic_node(state: AgentState) -> AgentState:
        if state.get("route") != "factual":
            return {"critic_passed": True}
        chunks = state.get("chunks", [])
        if not chunks:
            return {"critic_passed": False}
        answer_words = _content_words(state.get("answer", ""))
        chunk_words: set[str] = set()
        for chunk in chunks:
            chunk_words |= _content_words(chunk.page_content)
        passed = bool(answer_words & chunk_words)
        if passed:
            return {"critic_passed": True}
        logger.warning("critic: answer failed grounding check")
        fallback = (
            "I could not find a grounded answer to that question in the "
            "indexed documents. Try rephrasing, or ingest documents that "
            "cover the topic."
        )
        return {"critic_passed": False, "answer": fallback}

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", "critic")
    graph.add_edge("critic", END)
    return graph.compile()
