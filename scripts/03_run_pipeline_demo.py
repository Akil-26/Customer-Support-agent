"""
Batch sanity check for the current pipeline (agent.pipeline.run_agent)
against data/eval/golden_set.csv.

Distinct from scripts/02_test_agent.py, which spot-checks a handful of
hand-picked cases (multi-intent, thread-context, risk keywords) -- this
script runs the pipeline over a random sample of the 200-row golden set
and reports how often predicted intent/escalate match the hand labels.
NOT the real Step 6 evaluation harness -- just a fast sanity check before
building that.

Usage:
    python scripts/03_run_pipeline_demo.py            # 5 random golden rows
    python scripts/03_run_pipeline_demo.py 15          # 15 random golden rows
    python scripts/03_run_pipeline_demo.py --text "my package never arrived"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from agent.pipeline import run_agent
from agent.retriever import ReplyRetriever
from agent.config import GOLDEN_SET_CSV

PAIRS_PATH = "data/processed/amazon_pairs.csv"


def run_single(text: str, retriever: ReplyRetriever):
    print(f"\nCustomer message:\n  \"{text}\"\n")
    result = run_agent(text, retriever)
    _print_result(result)


def run_from_golden_set(n: int, retriever: ReplyRetriever):
    df = pd.read_csv(GOLDEN_SET_CSV)
    sample = df.sample(n=min(n, len(df)), random_state=None)

    correct_intent = 0
    correct_escalate = 0

    for _, row in sample.iterrows():
        print("=" * 70)
        print(f'Customer message:\n  "{row.customer_clean}"')
        print(f"  Gold intent   : {row.intent}   |   Gold escalate: {row.escalate}")

        result = run_agent(row.customer_clean, retriever)
        _print_result(result, indent="  ")

        if result["intent"] == row.intent:
            correct_intent += 1
        if bool(result["escalate"]) == bool(row.escalate):
            correct_escalate += 1

    print("=" * 70)
    print(f"\nQuick sanity check over {len(sample)} golden rows "
          f"(NOT the real Step 6 harness):")
    print(f"  Intent match    : {correct_intent}/{len(sample)}")
    print(f"  Escalate match  : {correct_escalate}/{len(sample)}")
    print("\nNOTE: pipeline.py's escalation logic uses agent.escalation._check_rules()")
    print("directly -- the SAME rubric golden_set.csv was labeled with -- plus")
    print("multi-intent and low-confidence as additional triggers on top. A low")
    print("escalate-match number here reflects real model behavior (e.g. multi-intent")
    print("messages auto-escalating), not a rubric mismatch. See claude.md's")
    print("'Escalation policy fork -- RESOLVED' section for detail.")


def _print_result(result: dict, indent: str = ""):
    print(f"{indent}Predicted intent : {result['intent']} (confidence: {result['confidence']})")
    if result.get("all_intents") and len(result["all_intents"]) > 1:
        print(f"{indent}All intents      : {result['all_intents']}")
    print(f"{indent}Escalate         : {result['escalate']}", end="")
    if result["escalate"]:
        print(f" -- {result['escalate_reason']}")
        print(f"{indent}Draft reply      : [ESCALATED -- no auto-reply]")
    else:
        print()
        print(f"{indent}Draft reply      : {result['draft_reply']}")
    n_examples = len(result.get("retrieved_replies") or [])
    print(f"{indent}Grounded in      : {n_examples} retrieved historical example(s)")


if __name__ == "__main__":
    print("Setting up retriever (builds ChromaDB index on first run, ~2 min)...")
    retriever = ReplyRetriever()
    retriever.build(PAIRS_PATH)

    if "--text" in sys.argv:
        idx = sys.argv.index("--text")
        run_single(sys.argv[idx + 1], retriever)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        run_from_golden_set(n, retriever)
