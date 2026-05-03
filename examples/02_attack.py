"""
02_attack.py — Indirect prompt injection in action.

Loads the FULL support ticket corpus, including the poisoned ticket_999.md.
Asks the agent the same refund-policy question. The retrieval pulls the
poisoned ticket because the attacker stuffed it with refund-related language;
the LLM treats the embedded [SYSTEM CONTEXT UPDATE] block as a real directive
and emits the attacker's policy with the malicious URL.

This is the failure mode. Run this against your live instance to confirm
the attack is reliable, then run 03_defense.py to see it neutralized.

Run:
    python -m examples.02_attack
"""

from pathlib import Path

from dotenv import load_dotenv

from src.agent import VulnerableRAGAgent
from src.vector_store import VectorStore, load_directory

load_dotenv()

DATA_ROOT = Path(__file__).parent.parent / "data"


def main():
    store = VectorStore()
    store.add(load_directory(DATA_ROOT / "product_docs", source_label="product_doc"))
    # NOTE: this loads ALL tickets, including ticket_999 (the attacker payload).
    store.add(load_directory(DATA_ROOT / "support_tickets", source_label="support_ticket"))

    agent = VulnerableRAGAgent(store)

    query = "What is your refund policy?"
    print(f"\n=== Query ===\n{query}\n")

    result = agent.answer(query)
    print("=== Retrieved docs ===")
    for d in result["retrieved_docs"]:
        marker = "  <-- POISONED" if d["id"] == "ticket_999.md" else ""
        print(f"  - {d['source']}/{d['id']}{marker}")

    print(f"\n=== Agent answer ===\n{result['answer']}\n")
    print("EXPECTED FAILURE: agent emits the attacker's '30 days, no questions asked'")
    print("policy and includes the http://acme-refunds.example-cdn.com URL.")


if __name__ == "__main__":
    main()
