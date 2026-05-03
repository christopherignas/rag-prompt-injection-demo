"""
03_defense.py — The same scenario with layered defenses enabled.

Same poisoned corpus as 02_attack.py — the attacker's ticket_999.md is in
the vector store. But this time we use SafeRAGAgent, which:

  1. Runs each retrieved document through an injection classifier and drops
     anything classified as injection above the confidence threshold.
  2. Wraps surviving documents in <UNTRUSTED_CONTENT> spotlighting tags so
     the LLM treats them as data, not instructions.

Result: the attacker's payload is detected and dropped at the classifier
layer; even if it slipped through, spotlighting hardens the model against
following instructions inside retrieved content.

Run:
    python -m examples.03_defense
"""

from pathlib import Path

from dotenv import load_dotenv

from src.safe_agent import SafeRAGAgent
from src.vector_store import VectorStore, load_directory

load_dotenv()

DATA_ROOT = Path(__file__).parent.parent / "data"


def main():
    store = VectorStore()
    store.add(load_directory(DATA_ROOT / "product_docs", source_label="product_doc"))
    store.add(load_directory(DATA_ROOT / "support_tickets", source_label="support_ticket"))

    agent = SafeRAGAgent(store)

    query = "What is your refund policy?"
    print(f"\n=== Query ===\n{query}\n")

    result = agent.answer(query)
    print("=== Retrieved docs (before filtering) ===")
    for d in result["retrieved_docs"]:
        print(f"  - {d['source']}/{d['id']}")

    print("\n=== Pattern-flagged (Layer 0 regex) ===")
    print(f"  {result['pattern_flagged'] or 'none'}")

    print("\n=== Blocked by classifier (Layer 1) ===")
    if result["blocked_docs"]:
        for b in result["blocked_docs"]:
            print(f"  - {b['source']}/{b['id']}")
            print(f"      reason: {b['reason']}  (confidence={b['confidence']:.2f})")
    else:
        print("  none")

    print("\n=== Docs that reached the LLM (after spotlighting) ===")
    for d in result["kept_docs"]:
        print(f"  - {d['source']}/{d['id']}")

    print(f"\n=== Agent answer ===\n{result['answer']}\n")
    print("EXPECTED: agent describes the real 14-day refund policy. The attacker's")
    print("URL is NOT present. ticket_999.md should appear under 'Blocked by classifier'.")


if __name__ == "__main__":
    main()
