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
- **The index is persisted to disk.** On ingest, the FAISS index
  (`faiss.write_index`) and chunk metadata (JSON) are saved to
  `INDEX_DIR` (default `data/index/`); on startup a persisted index
  is restored, so ingested documents survive restarts and the bundled
  docs are not re-added (no duplicates). A missing, corrupt, or
  dimension-mismatched store simply starts empty and falls back to
  the bundled docs.
- **The API has optional API-key auth.** Set `API_KEY` and clients
  must send it as an `X-API-Key` header on `/ask` and `/ingest`
  (`/health` stays open). With no key set the API runs open and logs
  a warning at startup — local demo mode only, not a hosted setup.

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

The server starts on http://localhost:8000. On startup it restores
the persisted index from `INDEX_DIR` if one exists, otherwise it
indexes the synthetic docs in `data/sample_docs/` (and persists them
for the next start).

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

# Ingest your own documents (add -H 'X-API-Key: <key>' if API_KEY is set)
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "doc1", "title": "My notes",
      "text": "The zebraquantum protocol synchronises widgets."}]}'

# With API_KEY set, protected routes require the header (401 otherwise)
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' -H 'X-API-Key: <key>' \
  -d '{"question": "What is RAG?"}'
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
| `INDEX_DIR` | Persisted index directory (default `data/index`) |
| `API_KEY` | Optional key for `/ask` + `/ingest` via `X-API-Key`; unset = open demo mode with a startup warning |
| `TOP_K` | Chunks retrieved per question (default `3`) |

## Tests

```bash
PYTHONPATH=src pytest -q
```

Tests run fully offline: they use the extractive fallback path, so no
API key is needed and no network call is made. Coverage: document
loading/splitting, FAISS retrieval relevance, router routing, the full
graph (grounding + general path), the API endpoints including
ingest-then-ask, FAISS persistence round-trips (ingest -> reload ->
ask finds the doc, plus corrupt/dimension-mismatch fallbacks), and
API-key auth in both modes (open when unset, 401 without/wrong key
when set). Test runs persist into a throwaway temp dir, never into
the repo's `data/index/`.

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
│   ├── api.py            # FastAPI app: /health, /ask, /ingest (+ API-key auth)
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

- Single-process index file: no locking, so run one instance per
  index directory; concurrent writers could corrupt the store.
- API-key auth is a shared secret, not real identity: no OAuth,
  no per-user permissions, no rate limiting, no HTTPS termination
  in this app (use a reverse proxy for any real exposure).
- The live LLM path has not been exercised with a real key — only
  the offline fallback is tested. A real-key run is still pending.
- Hashed embeddings capture keyword overlap, not deep semantics.
- The critic checks word overlap, not true factual correctness.
- The eval set is 6 hand-written questions — a smoke signal, not a
  benchmark.
