"""
Step 5 -- Baselines (trivial + simple), evaluated against data/eval/golden_set.csv.

Hiver's rubric requires "results vs. at least two baselines (a trivial one
and a simple one)" in the report. This script produces both, deliberately
WITHOUT any LLM call -- both baselines are fully deterministic and need no
GROQ_API_KEY, so this is also part of the fix for "<15 minute reproduction"
(a reviewer can see real baseline numbers with zero setup cost).

TRIVIAL baseline (the floor -- what you'd get with no real signal at all):
  - intent    : always the single most common candidate_intent across the
                full ~40k-row corpus (NOT computed from golden_set.csv
                itself, to avoid tuning the "trivial" baseline on the eval
                set -- see decision log).
  - reply     : always the single most common historical brand_reply_clean
                string across the full corpus (a generic, real Amazon
                reply, but the exact same one regardless of what the
                customer said).
  - escalate  : always False (never escalate).

SIMPLE baseline (a real but cheap signal -- no LLM, no fine-tuning):
  - intent    : 1-nearest-neighbor via TF-IDF cosine similarity against the
                full corpus (using the same candidate_intent_by_keywords
                heuristic labels golden_set.csv itself was seeded from --
                see golden_set_methodology.md).
  - reply     : the nearest neighbor's actual historical brand_reply_clean.
  - escalate  : rule-only (agent.escalation._check_rules), i.e. the same
                Stage-1 rubric the real pipeline uses, but with NO Stage-2
                LLM fallback for ambiguous cases -- those default to False.
                This isolates how much the LLM escalation step actually
                adds over pure keyword rules once Step 6 compares the two.

Neither baseline is grounded per-message the way the real pipeline's RAG +
LLM draft is -- that gap is exactly what Step 6's evaluation harness needs
to quantify, not just assert.

Outputs:
  - data/eval/baseline_predictions.csv  (per-row predictions, both baselines,
    reusable by Step 6 so the real pipeline's predictions can be added as
    a third column set for a fair side-by-side comparison)
  - reports/baseline_metrics.json       (summary metrics)
  - console: a formatted comparison table

Usage:
    python scripts/04_baselines.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agent.config import AMAZON_PAIRS_CSV, GOLDEN_SET_CSV
from agent.intents import candidate_intent_by_keywords
from agent.escalation import _check_rules

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
PREDICTIONS_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval", "baseline_predictions.csv"
)


def load_corpus() -> pd.DataFrame:
    print(f"Loading corpus: {AMAZON_PAIRS_CSV}")
    df = pd.read_csv(AMAZON_PAIRS_CSV)
    before = len(df)
    df = df[df["has_reply"] == True].copy()
    df = df.dropna(subset=["customer_clean", "brand_reply_clean"])
    df = df[df["customer_clean"].str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"  {before} rows -> {len(df)} usable (has_reply, non-empty)")

    print("  Tagging corpus with candidate_intent (keyword heuristic, same as golden_set)...")
    tqdm.pandas(desc="  candidate_intent")
    df["candidate_intent"] = df["customer_clean"].progress_apply(candidate_intent_by_keywords)
    return df


def build_trivial_baseline(corpus: pd.DataFrame) -> dict:
    trivial_intent = corpus["candidate_intent"].value_counts().idxmax()
    trivial_reply = corpus["brand_reply_clean"].value_counts().idxmax()
    print(f"\n[Trivial baseline]")
    print(f"  Always predicts intent   : {trivial_intent!r}")
    print(f"  Always predicts reply    : {trivial_reply[:100]!r}...")
    print(f"  Always predicts escalate : False")
    return {"intent": trivial_intent, "reply": trivial_reply, "escalate": False}


def build_simple_baseline(corpus: pd.DataFrame):
    print(f"\n[Simple baseline] Fitting TF-IDF on {len(corpus)} corpus messages...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2)
    corpus_vectors = vectorizer.fit_transform(corpus["customer_clean"])

    print("  Fitting 1-NN index (cosine)...")
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(corpus_vectors)

    return vectorizer, nn


def predict_simple(message: str, vectorizer, nn, corpus: pd.DataFrame) -> dict:
    vec = vectorizer.transform([message])
    _, idx = nn.kneighbors(vec, n_neighbors=1)
    neighbor = corpus.iloc[idx[0][0]]
    intent = neighbor["candidate_intent"]

    rule_hit = _check_rules(message, intent)
    escalate = bool(rule_hit["escalate"]) if rule_hit else False
    reason = rule_hit["reason"] if rule_hit else "No escalation rule fired (simple baseline has no LLM fallback)."

    return {
        "intent": intent,
        "reply": neighbor["brand_reply_clean"],
        "escalate": escalate,
        "escalate_reason": reason,
    }


def evaluate(golden: pd.DataFrame, intent_col: str, escalate_col: str, label: str) -> dict:
    intent_acc = accuracy_score(golden["intent"], golden[intent_col])

    escalate_p, escalate_r, escalate_f1, _ = precision_recall_fscore_support(
        golden["escalate"], golden[escalate_col], average="binary", zero_division=0
    )
    escalate_acc = accuracy_score(golden["escalate"], golden[escalate_col])

    print(f"\n[{label}] vs golden_set.csv (n={len(golden)})")
    print(f"  Intent accuracy          : {intent_acc:.3f}")
    print(f"  Escalate accuracy        : {escalate_acc:.3f}")
    print(f"  Escalate precision       : {escalate_p:.3f}")
    print(f"  Escalate recall          : {escalate_r:.3f}")
    print(f"  Escalate F1              : {escalate_f1:.3f}")

    return {
        "intent_accuracy": round(intent_acc, 4),
        "escalate_accuracy": round(escalate_acc, 4),
        "escalate_precision": round(escalate_p, 4),
        "escalate_recall": round(escalate_r, 4),
        "escalate_f1": round(escalate_f1, 4),
    }


def main():
    corpus = load_corpus()
    golden = pd.read_csv(GOLDEN_SET_CSV)
    print(f"\nLoaded golden set: {len(golden)} rows")

    trivial = build_trivial_baseline(corpus)
    vectorizer, nn = build_simple_baseline(corpus)

    print(f"\nRunning simple baseline over {len(golden)} golden-set messages...")
    simple_predictions = [
        predict_simple(msg, vectorizer, nn, corpus)
        for msg in tqdm(golden["customer_clean"], desc="  1-NN retrieval")
    ]

    golden["trivial_intent"] = trivial["intent"]
    golden["trivial_reply"] = trivial["reply"]
    golden["trivial_escalate"] = trivial["escalate"]

    golden["simple_intent"] = [p["intent"] for p in simple_predictions]
    golden["simple_reply"] = [p["reply"] for p in simple_predictions]
    golden["simple_escalate"] = [p["escalate"] for p in simple_predictions]
    golden["simple_escalate_reason"] = [p["escalate_reason"] for p in simple_predictions]

    trivial_metrics = evaluate(golden, "trivial_intent", "trivial_escalate", "TRIVIAL baseline")
    simple_metrics = evaluate(golden, "simple_intent", "simple_escalate", "SIMPLE baseline")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = {
        "n_golden_rows": len(golden),
        "n_corpus_rows": len(corpus),
        "trivial": {
            "fixed_intent": trivial["intent"],
            "fixed_reply": trivial["reply"],
            **trivial_metrics,
        },
        "simple": simple_metrics,
    }
    metrics_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    golden.to_csv(PREDICTIONS_OUT, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (trivial vs simple, on the 200-row golden set)")
    print("=" * 70)
    print(f"{'Metric':<25}{'Trivial':>15}{'Simple':>15}")
    for key in ["intent_accuracy", "escalate_accuracy", "escalate_precision", "escalate_recall", "escalate_f1"]:
        print(f"{key:<25}{trivial_metrics[key]:>15.3f}{simple_metrics[key]:>15.3f}")
    print("=" * 70)
    print(f"\nSaved predictions -> {PREDICTIONS_OUT}")
    print(f"Saved metrics     -> {metrics_path}")
    print("\nNext: run the real agent pipeline over the same golden set (Step 6) and")
    print("add a third column set to baseline_predictions.csv for a fair 3-way comparison.")


if __name__ == "__main__":
    main()
