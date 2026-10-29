"""FastAPI application: /health, /ask, /ingest.

Thin by design — all agent logic lives in :mod:`service`. Request and
response shapes are Pydantic models so invalid input is rejected with
a clear 422 before any agent work runs.

Authentication: if the ``API_KEY`` environment variable is set, ``/ask``
and ``/ingest`` require it as an ``X-API-Key`` header (401 otherwise);
``/health`` always stays open. If it is unset, the API runs open and
logs a warning at startup — that mode is for local demos only, not
for a hosted service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .logging_config import get_logger
from .service import AssistantService

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(service: AssistantService | None = None) -> FastAPI:
    """Build the FastAPI app around ``service`` (fresh one if omitted)."""
    svc = service or AssistantService()

    def verify_api_key(provided: str | None = Depends(_api_key_header)) -> None:
        """Reject protected calls when auth is enabled and the key is wrong."""
        if not svc.settings.auth_enabled:
            return
        if provided is None or provided != svc.settings.api_key:
            raise HTTPException(
                status_code=401, detail="missing or invalid X-API-Key header"
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Load the index once at startup; warn when running open."""
        if not svc.settings.auth_enabled:
            logger.warning(
                "API_KEY is not set — /ask and /ingest run OPEN. "
                "Local demo mode only; set API_KEY before exposing this API."
            )
        added = svc.load_sample_docs()
        logger.info(
            "startup: %d chunks indexed (%d newly added)",
            svc.docs_indexed,
            added,
        )
        yield

    app = FastAPI(
        title="agentic-rag-assistant",
        description=(
            "Personal portfolio project: a multi-agent RAG assistant "
            "(router -> retrieval -> answer -> critic) over indexed documents."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness check; also reports index size and LLM configuration."""
        return HealthResponse(
            chunks_indexed=svc.docs_indexed,
            llm_configured=svc.settings.llm_configured,
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        dependencies=[Depends(verify_api_key)],
    )
    def ask(payload: AskRequest) -> AskResponse:
        """Answer one question through the agent graph."""
        try:
            result = svc.ask(payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AskResponse(**result)

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        dependencies=[Depends(verify_api_key)],
    )
    def ingest(payload: IngestRequest) -> IngestResponse:
        """Index new documents (persisted; they survive restarts)."""
        added, total = svc.ingest_texts(
            [(d.id, d.title, d.text) for d in payload.documents]
        )
        return IngestResponse(chunks_added=added, total_chunks=total)

    return app


class AskRequest(BaseModel):
    """A question for the assistant."""

    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    """An answer plus provenance, so grounding is auditable."""

    answer: str
    sources: list[str] = []
    route: str
    critic_passed: bool
    llm_used: bool


class IngestDocument(BaseModel):
    """One document to index."""

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    """Documents to add to the index."""

    documents: list[IngestDocument] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    """Result of an ingest call."""

    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    """Liveness plus index/LLM configuration status."""

    status: str = "ok"
    chunks_indexed: int
    llm_configured: bool


# Default app for ``uvicorn agentic_rag_assistant.api:app`` and for the
# existing test suite. Tests needing custom settings use create_app().
app = create_app()
