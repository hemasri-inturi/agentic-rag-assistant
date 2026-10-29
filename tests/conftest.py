"""Test setup: put src/ on sys.path and force offline demo mode."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep the suite hermetic: unless the runner explicitly set INDEX_DIR,
# the shared app service persists into a throwaway temp dir, never into
# the repo's data/index/ (which is the runtime default). This runs at
# conftest import time, before test modules import the app.
os.environ.setdefault(
    "INDEX_DIR", tempfile.mkdtemp(prefix="rag-test-index-")
)


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch):
    """Ensure no real LLM key or API key leaks into tests."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    yield
