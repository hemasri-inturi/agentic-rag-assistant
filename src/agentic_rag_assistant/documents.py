"""Document loading and chunking.

Sample documents are synthetic FAQ texts written for this project.
Loading uses LangChain's text splitter so chunking behaviour matches
the wider LangChain ecosystem; the splitter itself needs no API key.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .logging_config import get_logger

logger = get_logger(__name__)


def load_markdown_files(directory: str | Path) -> list[Document]:
    """Load every ``*.md`` file in ``directory`` as a Document.

    Each document carries ``source`` (filename) and ``title`` (the first
    markdown heading, or the filename stem) as metadata.
    """
    docs: list[Document] = []
    path = Path(directory)
    if not path.is_dir():
        logger.warning("document directory not found: %s", path)
        return docs
    for file in sorted(path.glob("*.md")):
        text = file.read_text(encoding="utf-8")
        title = file.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file.name, "title": title},
            )
        )
    logger.info("loaded %d documents from %s", len(docs), path)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("split %d documents into %d chunks", len(docs), len(chunks))
    return chunks
