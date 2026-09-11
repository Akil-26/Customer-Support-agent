"""
Script: Test the thread-aware agent pipeline
=============================================
Tests with and without thread context to show the difference.

Usage:
  python scripts/02_test_agent.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.retriever import ReplyRetriever
from agent.pipeline  import run_agent

PAIRS_PATH = "data/processed/amazon_pairs_with_context.csv"

# Format: (category, message, thread_context)
TEST_CASES = [
    # ── Without thread context ──────────────────────────────────────────
    ("SINGLE", "my order hasn't arrived and it's been 5 days past the delivery date", ""),
    ("SINGLE", "my Echo Show keeps disconnecting from wifi every few hours", ""),
    ("SINGLE", "why was I charged twice for the same order last month", ""),
    ("SINGLE", "I can't log into my Amazon account, forgot my password", ""),

    # ── WITH thread context — resolves vague references ─────────────────
    (
        "WITH_CONTEXT",
        "I want to return this item, it arrived completely broken",
        "Customer: My Kindle Fire HD stopped working after 2 days\nAmazon: Sorry to hear that! Can you tell us more about the issue?"
    ),
    (
        "WITH_CONTEXT",
        "still no update on this",
        "Customer: I ordered a laptop 2 weeks ago and it never arrived\nAmazon: We're sorry! Can you DM us your order number?"
    ),
    (
        "WITH_CONTEXT",
        "yes the charge is still there",
        "Customer: I was charged twice for order #12345\nAmazon: We're looking into this, can you confirm the charge is still showing?"
    ),

    # ── Multi-intent — should escalate ──────────────────────────────────
    ("MULTI",  "my order arrived broken AND you charged me twice for it", ""),
    ("MULTI",  "I want a refund but I also can't log into my account", ""),

    # ── High-risk — should escalate ─────────────────────────────────────
    ("RISK",   "I can't log into my account, someone hacked it", ""),
    ("RISK",   "I'm going to sue Amazon if this isn't resolved", ""),
]


def main():
    print("Setting up retriever...")
    retriever = ReplyRetriever()
    retriever.build(PAIRS_PATH)

    print("\n" + "=" * 65)
    print("AGENT TEST RUN — Thread-Aware Flow")
    print("=" * 65)

    for category, message, thread_context in TEST_CASES:
        print(f"\n[{category}]")
        if thread_context:
            print(f"  Context  : {thread_context.replace(chr(10), ' | ')}")
        print(f"  Message  : {message}")

        result = run_agent(message, retriever, thread_context=thread_context)

        print(f"  Intent   : {result['intent']} ({result['confidence']})")
        if result['all_intents'] and len(result['all_intents']) > 1:
            print(f"  All      : {result['all_intents']}")
        print(f"  Escalate : {result['escalate']}")
        if result['escalate']:
            print(f"  Reason   : {result['escalate_reason']}")
            print(f"  Reply    : [ESCALATED — no auto-reply]")
        else:
            print(f"  Reply    : {result['draft_reply']}")
        print("-" * 65)


if __name__ == "__main__":
    main()
