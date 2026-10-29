# Assistant FAQ (synthetic sample data)

This is a synthetic document written for this portfolio project. It is
not copied from any source.

## What does this assistant do?

This assistant answers questions from documents that have been indexed
into it. It uses a router agent to decide whether a question needs
retrieval, a retrieval agent to search a FAISS vector index, an answer
agent to produce the answer, and a critic step that checks the answer
is grounded in the retrieved chunks.

## How do you run this assistant?

Start the API server with uvicorn, then send a POST request to the
/ask endpoint with a JSON body containing a question. The /health
endpoint reports whether the service is up and how many chunks are
indexed. The /ingest endpoint adds new documents to the index.

## What happens without an API key?

Without an OPENAI_API_KEY, the assistant runs in demo fallback mode:
answers are extracted from the retrieved chunks instead of generated
by a language model. The API response marks this with llm_used set to
false, so the mode is always visible to the caller.
