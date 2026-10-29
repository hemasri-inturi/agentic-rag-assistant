"""Entry point: ``python -m agentic_rag_assistant.main``."""

from __future__ import annotations

import uvicorn

from .api import app


def main() -> None:
    """Serve the API on port 8000."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
