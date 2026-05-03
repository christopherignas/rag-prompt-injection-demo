"""
The defended RAG agent. Wraps the vulnerable agent with both layers of defense:
1. Injection classifier filters retrieved docs before they reach the LLM.
2. Spotlighting wraps surviving docs in untrusted-content delimiters.

In production, additional layers would include: fine-tuned classifier instead
of LLM call, rate limits + monitoring on detection events, content provenance
signing, and retrieval source trust scoring. Out of scope for the demo.
"""

from __future__ import annotations

import os
from openai import OpenAI

from src.defenses import (
    InjectionClassifier,
    SPOTLIGHTED_SYSTEM_PROMPT,
    quick_pattern_check,
    spotlight,
)
from src.vector_store import VectorStore

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


class SafeRAGAgent:
    """RAG with layered defenses against indirect prompt injection."""

    def __init__(
        self,
        store: VectorStore,
        client: OpenAI | None = None,
        classifier_threshold: float = 0.7,
    ):
        self.store = store
        self.client = client or OpenAI()
        self.classifier = InjectionClassifier(client=self.client)
        self.threshold = classifier_threshold

    def answer(self, query: str, k: int = 3) -> dict:
        retrieved = self.store.retrieve(query, k=k)

        # Layer 0 (free): regex pre-filter catches obvious payloads.
        pattern_flagged = [d for d in retrieved if quick_pattern_check(d)]

        # Layer 1: classifier filters everything (including pattern-flagged docs,
        # so we get a confidence score for the trace).
        safe_docs, classifier_results = self.classifier.filter(retrieved, self.threshold)

        # Layer 2: spotlight whatever survives.
        context = spotlight(safe_docs)
        system_msg = SPOTLIGHTED_SYSTEM_PROMPT.format(context=context)

        # Generate response.
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
            "kept_docs": [{"id": d.id, "source": d.source} for d in safe_docs],
            "blocked_docs": [
                {
                    "id": d.id,
                    "source": d.source,
                    "reason": r.reason,
                    "confidence": r.confidence,
                }
                for d, r in zip(retrieved, classifier_results)
                if r.is_injection and r.confidence >= self.threshold
            ],
            "pattern_flagged": [d.id for d in pattern_flagged],
        }
