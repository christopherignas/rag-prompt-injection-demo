"""
Tiny in-memory vector store. No external DB dependency — keeps the demo
focused on the prompt-injection attack rather than infrastructure.

Embeddings via OpenAI text-embedding-3-small. Retrieval via cosine similarity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


@dataclass
class Document:
    """A single retrievable chunk with its metadata."""
    id: str          # filename or unique identifier
    source: str      # category, e.g. "product_doc" or "support_ticket"
    text: str
    embedding: np.ndarray | None = None


class VectorStore:
    """Embed + cosine-similarity retrieval. Tiny on purpose."""

    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.docs: list[Document] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Batch-embed a list of strings. Returns (N, dim) numpy array."""
        response = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in response.data])

    def add(self, docs: Iterable[Document]) -> None:
        """Embed and store a batch of documents."""
        docs = list(docs)
        if not docs:
            return
        embeddings = self._embed([d.text for d in docs])
        for d, emb in zip(docs, embeddings):
            d.embedding = emb
        self.docs.extend(docs)

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Return the top-k documents most similar to the query."""
        if not self.docs:
            return []
        query_emb = self._embed([query])
        doc_embs = np.vstack([d.embedding for d in self.docs])
        scores = cosine_similarity(query_emb, doc_embs)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in top_idx]


def load_directory(root: Path | str, source_label: str) -> list[Document]:
    """Load all .md files in a directory as Documents."""
    root = Path(root)
    docs = []
    for path in sorted(root.glob("*.md")):
        docs.append(Document(
            id=path.name,
            source=source_label,
            text=path.read_text(encoding="utf-8"),
        ))
    return docs
