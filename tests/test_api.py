"""Tests for the FastAPI endpoints (offline fallback mode)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_rag_assistant.api import app

client = TestClient(app)


def test_health_reports_index():
    """Health returns ok + chunk count + llm_configured=false offline."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is False
    assert body["chunks_indexed"] >= 0


def test_ask_sample_question():
    """A question about the bundled docs gets a grounded answer."""
    response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "factual"
    assert body["critic_passed"] is True
    assert body["llm_used"] is False
    assert "retrieval" in body["answer"].lower()


def test_ask_rejects_empty_question():
    """An empty question is rejected by request validation."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ingest_then_ask_finds_new_doc():
    """Ingested docs are searchable immediately (unique token test)."""
    ingest = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "id": "test-doc",
                    "title": "Zebraquantum doc",
                    "text": (
                        "The zebraquantum protocol synchronises widgets "
                        "across devices using a shared ledger."
                    ),
                }
            ]
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    # Query with several distinctive terms from the new doc so the test
    # measures retrieval of that doc, not generic-word overlap.
    response = client.post(
        "/ask",
        json={"question": "zebraquantum protocol synchronises widgets"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "zebraquantum" in body["answer"].lower()
    assert "Zebraquantum doc" in body["sources"]
