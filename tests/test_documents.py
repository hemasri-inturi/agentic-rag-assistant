"""Tests for document loading and chunking."""

from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag_assistant.documents import (
    load_markdown_files,
    split_documents,
)


def test_load_markdown_files(tmp_path):
    """Markdown files load with source + title metadata."""
    (tmp_path / "a.md").write_text("# Alpha Doc\n\nHello world.", encoding="utf-8")
    (tmp_path / "b.md").write_text("No heading here.", encoding="utf-8")
    docs = load_markdown_files(tmp_path)
    assert len(docs) == 2
    assert docs[0].metadata["title"] == "Alpha Doc"
    assert docs[0].metadata["source"] == "a.md"
    assert docs[1].metadata["title"] == "b"


def test_load_missing_directory_returns_empty(tmp_path):
    """A missing directory is a warning + empty list, not a crash."""
    assert load_markdown_files(tmp_path / "does-not-exist") == []


def test_split_documents_chunks_long_text():
    """Long text splits into multiple overlapping chunks."""
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_documents(
        [Document(page_content=text, metadata={"title": "t"})],
        chunk_size=100,
        chunk_overlap=20,
    )
    assert len(chunks) > 1
    assert all(c.page_content.strip() for c in chunks)
    # Metadata survives splitting.
    assert all(c.metadata["title"] == "t" for c in chunks)
