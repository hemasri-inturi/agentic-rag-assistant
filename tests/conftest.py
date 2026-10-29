"""Test setup: put src/ on sys.path and force offline demo mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch):
    """Ensure no real LLM key leaks into tests; fallback path is tested."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield
