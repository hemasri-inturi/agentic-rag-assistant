"""LLM access via an OpenAI-compatible HTTP API — and the offline fallback.

Only this module ever touches the network, and only when a key is
configured. The chat call uses plain ``httpx`` against
``OPENAI_BASE_URL`` so any OpenAI-compatible endpoint works
(OpenAI, a local server, or another provider's compatible API).

When no key is configured, :func:`extractive_answer` answers from the
retrieved chunks instead. That path is always labelled as a demo
fallback in the API response (``llm_used=False``) so a reader can never
mistake it for a model-generated answer.
"""

from __future__ import annotations

import httpx
from langchain_core.documents import Document

from .config import Settings
from .logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY from the documents below. "
    "If the documents do not contain the answer, say so plainly instead "
    "of guessing. Keep the answer concise."
)


def extractive_answer(question: str, chunks: list[Document]) -> str:
    """Build a grounded demo answer from retrieved chunks (no LLM).

    The answer quotes the most relevant chunk, so it is grounded by
    construction. The caller labels this path as a demo fallback.
    """
    if not chunks:
        return (
            "I could not find anything about that in the indexed "
            "documents. Try rephrasing the question, or ingest documents "
            "that cover the topic."
        )
    top = chunks[0].page_content.strip().replace("\n", " ")
    excerpt = top if len(top) <= 600 else top[:597] + "..."
    return (
        "Based on the retrieved documents: " + excerpt + " "
        "[Demo fallback: no LLM API key is configured, so this answer "
        "is extracted from the retrieved text rather than generated.]"
    )


def generate_answer(
    settings: Settings,
    question: str,
    chunks: list[Document],
) -> tuple[str, bool]:
    """Generate an answer; returns ``(answer, llm_used)``.

    Falls back to the extractive answer on any LLM failure, and logs
    the failure instead of leaking provider error details to the user.
    """
    if not settings.llm_configured:
        return extractive_answer(question, chunks), False
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Documents:\n{context}\n\nQuestion: {question}",
        },
    ]
    try:
        response = httpx.post(
            settings.openai_base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.model_name,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return str(content).strip(), True
    except Exception as exc:  # noqa: BLE001 - fall back, never crash /ask
        logger.warning("LLM call failed, using extractive fallback: %s", exc)
        return extractive_answer(question, chunks), False
