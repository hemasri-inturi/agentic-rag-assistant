"""FastAPI application: /health, /ask, /ingest.

Thin by design — all agent logic lives in :mod:`service`. Request and
response shapes are Pydantic models so invalid input is rejected with
a clear 422 before any agent work runs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .logging_config import get_logger
from .service import AssistantService

logger = get_logger(__name__)
service = AssistantService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the bundled synthetic docs once at startup."""
    added = service.load_sample_docs()
    logger.info("startup: indexed %d sample chunks", added)
    yield


app = FastAPI(
    title="agentic-rag-assistant",
    description=(
        "Personal portfolio project: a multi-agent RAG assistant "
        "(router -> retrieval -> answer -> critic) over indexed documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check; also reports index size and LLM configuration."""
    return HealthResponse(
        chunks_indexed=service.docs_indexed,
        llm_configured=service.settings.llm_configured,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Answer one question through the agent graph."""
    try:
        result = service.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AskResponse(**result)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    """Index new documents (in-memory; lost on restart by design)."""
    added, total = service.ingest_texts(
        [(d.id, d.title, d.text) for d in payload.documents]
    )
    return IngestResponse(chunks_added=added, total_chunks=total)
