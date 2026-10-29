# RAG Concepts FAQ (synthetic sample data)

This is a synthetic document written for this portfolio project. It is
not copied from any source.

## What is RAG?

Retrieval-Augmented Generation, usually called RAG, is a technique that
combines retrieval with generation. First, a retriever finds the most
relevant documents for a question. Then, a language model generates an
answer grounded in those retrieved documents, instead of answering only
from what it memorised during training.

## What is an embedding?

An embedding is a numeric vector that represents the meaning of a piece
of text. Texts with similar meanings have embeddings that are close
together, which is what makes similarity search possible.

## What is a vector database?

A vector database stores embeddings and supports fast similarity
search over them. FAISS is a popular open-source library for this: it
indexes vectors so the nearest neighbours of a query vector can be
found quickly.

## What is chunking?

Chunking is the practice of splitting long documents into smaller
chunks before indexing them. Retrieval works better on chunks than on
whole documents, because a short chunk is more likely to be tightly
focused on one topic.
