"""
The vulnerable RAG agent. Retrieves context naively, concatenates everything
into the prompt, and asks the LLM to answer.

This is the *unprotected* version — see safe_agent.py for the defended one.
"""

from __future__ import annotations

import os
from openai import OpenAI

from src.vector_store import Document, VectorStore

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# A deliberately permissive system prompt — the kind a junior engineer
# might write before learning about indirect prompt injection.
NAIVE_SYSTEM_PROMPT = """\
You are a helpful customer support assistant. Use the following retrieved
documents to answer the user's question accurately and concisely.

Retrieved context:
{context}
"""


class VulnerableRAGAgent:
    """Plain RAG: retrieve, stuff into prompt, complete. No defenses."""

    def __init__(self, store: VectorStore, client: OpenAI | None = None):
        self.store = store
        self.client = client or OpenAI()

    def _format_context(self, docs: list[Document]) -> str:
        """Concatenate retrieved docs verbatim. THIS IS THE BUG — no isolation."""
        return "\n\n---\n\n".join(
            f"Source: {d.source}/{d.id}\n\n{d.text}" for d in docs
        )

    def answer(self, query: str, k: int = 3) -> dict:
        """Retrieve k docs, build prompt, call the LLM, return result + trace."""
        retrieved = self.store.retrieve(query, k=k)
        context = self._format_context(retrieved)
        system_msg = NAIVE_SYSTEM_PROMPT.format(context=context)

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )

        return {
            "answer": response.choices[0].message.content,
            "retrieved_docs": [{"id": d.id, "source": d.source} for d in retrieved],
        }
