# agentic-rag-assistant

> **Personal portfolio project.** Built to demonstrate production-style
> engineering on a small scale: API design, a multi-agent flow,
> retrieval, evaluation, tests and CI. It makes no claim about any
> client, employer, production deployment, user count or business
> metric, and the bundled documents are synthetic texts written for
> this project.

A multi-agent RAG (Retrieval-Augmented Generation) assistant. Ask a
question, and it routes the question, retrieves the most relevant
document chunks from a FAISS index, answers from those chunks, and
runs a critic step that checks the answer is actually grounded in what
was retrieved.

## Architecture

```
                         ┌──────────┐
  POST /ask ─────────────►│  Router  │── general ──► general answer
                         └────┬─────┘
                              │ factual
                         ┌────▼─────┐
                         │ Retrieval│── FAISS + hashed embeddings
                         └────┬─────┘
                              │
                         ┌────▼─────┐
                         │  Answer  │── OpenAI-compatible LLM, or
                         └────┬─────┘   offline extractive fallback
                              │
                         ┌────▼─────┐
                         │  Critic  │── grounding check; replaces
                         └────┬─────┘   ungrounded answers honestly
                              │
                         response (answer + sources + flags)
```

Design choices, stated plainly:

- **Router is rule-based, not LLM-based.** Routing greetings and meta
  questions needs no model call, is fully testable offline, and its
  decisions are explainable.
- **Retrieval uses deterministic hashed embeddings** instead of a
  hosted embedding API, so the whole pipeline except the optional LLM
  call runs offline. This is a documented simplification, not a claim
  of state-of-the-art retrieval quality.
- **The critic is a grounding check, not a guarantee.** It verifies
  content-word overlap between answer and retrieved chunks and
  replaces ungrounded answers with an honest "not found" reply.
- **The index is in-memory.** Ingested documents are lost on restart.
  Persistence (e.g. a persisted FAISS index or a vector database) is
  the natural next step and is listed under Limitations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; only needed for a real LLM key
```

Requires Python 3.11+.

## Run

```bash
PYTHONPATH=src python -m agentic_rag_assistant.main
# or: uvicorn agentic_rag_assistant.api:app --port 8000  (with PYTHONPATH=src)
```

The server starts on http://localhost:8000. On startup it indexes the
synthetic docs in `data/sample_docs/`.

With Docker:

```bash
docker compose up --build
```

## API examples

```bash
# Health: index size + whether an LLM key is configured
curl http://localhost:8000/health

# Ask a question (offline demo fallback unless OPENAI_API_KEY is set)
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is RAG?"}'

# Ingest your own documents
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "doc1", "title": "My notes",
      "text": "The zebraquantum protocol synchronises widgets."}]}'
```

Response shape for `/ask`:

```json
{
  "answer": "Based on the retrieved documents: ...",
  "sources": ["RAG Concepts FAQ (synthetic sample data)"],
  "route": "factual",
  "critic_passed": true,
  "llm_used": false
}
```

`llm_used: false` means the answer was extracted from retrieved text
in demo fallback mode (no API key configured) — never a model answer
presented as anything else.

## Configuration

All configuration comes from environment variables (see
`.env.example`); no key is ever hardcoded:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Optional LLM key; unset = offline demo fallback |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint (default `https://api.openai.com/v1`) |
| `MODEL_NAME` | Chat model name (default `gpt-4o-mini`) |
| `DATA_DIR` | Startup docs directory (default `data/sample_docs`) |
| `TOP_K` | Chunks retrieved per question (default `3`) |

## Tests

```bash
PYTHONPATH=src pytest -q
```

Tests run fully offline: they use the extractive fallback path, so no
API key is needed and no network call is made. Coverage: document
loading/splitting, FAISS retrieval relevance, router routing, the full
graph (grounding + general path), and the API endpoints including
ingest-then-ask.

## Evaluation

```bash
PYTHONPATH=src python evals/run_eval.py
```

`evals/` holds a tiny sample set (`eval_set.json`, 6 questions) and a
script that scores keyword recall against the answer plus the critic's
grounding verdict. These scores are a development signal on a tiny
sample, not a benchmark claim.

## Project structure

```
agentic-rag-assistant/
├── src/agentic_rag_assistant/
│   ├── api.py            # FastAPI app: /health, /ask, /ingest
│   ├── service.py        # Service layer (store + graph + ingestion)
│   ├── graph.py          # LangGraph: router -> retrieval -> answer -> critic
│   ├── retriever.py      # FAISS vector store
│   ├── embeddings.py     # Deterministic hashed embedder (offline)
│   ├── llm.py            # OpenAI-compatible client + extractive fallback
│   ├── documents.py      # Markdown loading + LangChain chunking
│   ├── config.py         # Env-only settings (no hardcoded secrets)
│   ├── logging_config.py # Logging setup
│   └── main.py           # Uvicorn entry point
├── data/sample_docs/     # Synthetic FAQ docs (written for this project)
├── tests/                # pytest suite (offline)
├── evals/                # Tiny eval set + scoring script
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Limitations (honest list)

- In-memory index only; restart loses ingested documents.
- Hashed embeddings capture keyword overlap, not deep semantics.
- The critic checks word overlap, not true factual correctness.
- The eval set is 6 hand-written questions — a smoke signal, not a
  benchmark.
- No authentication on the API: this is a portfolio demo, not a
  hosted service.
