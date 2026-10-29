"""Configuration from environment variables only.

No secret is ever hardcoded. The LLM settings are optional: when
``OPENAI_API_KEY`` is unset, the assistant runs in a clearly-labelled
offline demo mode (extractive answers from retrieved chunks) so the
service and the test suite work with no key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read once from the environment."""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    data_dir: str = "data/sample_docs"
    index_dir: str = "data/index"
    api_key: str = ""
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_dim: int = 384

    @property
    def llm_configured(self) -> bool:
        """True only when a key is actually supplied."""
        return bool(self.openai_api_key)

    @property
    def auth_enabled(self) -> bool:
        """True only when an API key is actually supplied."""
        return bool(self.api_key)


def load_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        model_name=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        data_dir=os.environ.get("DATA_DIR", "data/sample_docs"),
        index_dir=os.environ.get("INDEX_DIR", "data/index"),
        api_key=os.environ.get("API_KEY", ""),
        top_k=int(os.environ.get("TOP_K", "3")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
    )
