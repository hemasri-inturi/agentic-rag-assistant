"""Tests for API-key auth on /ask and /ingest (offline, no keys)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_rag_assistant.api import create_app
from agentic_rag_assistant.config import Settings
from agentic_rag_assistant.service import AssistantService


def _client(tmp_path, api_key: str) -> TestClient:
    """Build an isolated app (own index dir, own service)."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    settings = Settings(
        data_dir=str(docs_dir),
        index_dir=str(tmp_path / "index"),
        api_key=api_key,
    )
    return TestClient(create_app(AssistantService(settings=settings)))


def test_auth_disabled_is_open(tmp_path):
    """With no API_KEY set, /ask and /ingest work without a header."""
    client = _client(tmp_path, api_key="")
    assert client.get("/health").status_code == 200
    assert (
        client.post("/ask", json={"question": "hello"}).status_code == 200
    )
    assert (
        client.post(
            "/ingest",
            json={
                "documents": [
                    {"id": "d1", "title": "T", "text": "hello world"}
                ]
            },
        ).status_code
        == 200
    )


def test_auth_enabled_rejects_missing_and_wrong_keys(tmp_path):
    """With API_KEY set, protected routes need the exact key."""
    client = _client(tmp_path, api_key="test-key-123")
    # Health stays open.
    assert client.get("/health").status_code == 200
    # Missing key -> 401 on both protected routes.
    assert client.post("/ask", json={"question": "hello"}).status_code == 401
    assert (
        client.post(
            "/ingest",
            json={
                "documents": [
                    {"id": "d1", "title": "T", "text": "hello world"}
                ]
            },
        ).status_code
        == 401
    )
    # Wrong key -> 401.
    wrong = {"X-API-Key": "nope"}
    assert (
        client.post("/ask", json={"question": "hello"}, headers=wrong).status_code
        == 401
    )
    # Correct key -> 200.
    right = {"X-API-Key": "test-key-123"}
    assert (
        client.post("/ask", json={"question": "hello"}, headers=right).status_code
        == 200
    )
    assert (
        client.post(
            "/ingest",
            json={
                "documents": [
                    {"id": "d1", "title": "T", "text": "hello world"}
                ]
            },
            headers=right,
        ).status_code
        == 200
    )
