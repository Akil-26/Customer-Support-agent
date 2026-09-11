"""
Script: Label ChromaDB pairs with intents using the classifier
==============================================================
Runs the intent classifier over all 40,324 pairs in amazon_pairs.csv
and saves a labeled version: data/processed/amazon_pairs_labeled.csv

This enables intent-filtered RAG retrieval — instead of pure semantic
similarity across all 40k pairs, we retrieve from the same intent bucket
as the incoming message. This gives much more relevant historical replies.

Run once — takes ~20 mins (40k Groq API calls with rate limiting).
Output is committed to repo so it doesn't need to be re-run.

Usage:
  python scripts/03_label_pairs.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

from agent.intents import INTENTS, INTENT_NAMES

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "openai/gpt-oss-120b"

PAIRS_PATH   = Path("data/processed/amazon_pairs.csv")
OUT_PATH     = Path("data/processed/amazon_pairs_labeled.csv")
BATCH_SIZE   = 1       # one at a time — Groq free tier rate limits
SLEEP_SEC    = 0.3     # pause between calls to avoid rate limit errors


def build_classifier_prompt() -> str:
    intent_lines = "\n".join(
        f"{i+1}. {intent['name']}: {intent['description'][:90]}"
        for i, intent in enumerate(INTENTS)
    )
    examples = "\n".join(
        f'  "{i["examples"][0][:80]}" -> {i["name"]}'
        for i in INTENTS
    )
    return (
        "Classify Amazon customer support tweets into exactly one intent.\n\n"
        f"Intents:\n{intent_lines}\n\n"
        f"Examples:\n{examples}\n\n"
        "Rules:\n"
        "- Pick exactly one intent. Use 'other' only if truly nothing fits.\n"
        'Output JSON only: {"intent": "<name>"}'
    )


SYSTEM_PROMPT = build_classifier_prompt()


def classify_one(message: str) -> str:
    """Classify a single message. Returns intent slug."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=50,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f'Tweet: "{message[:200]}"'},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract JSON if wrapped in text
        import re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)
        intent = parsed.get("intent", "other")
        return intent if intent in INTENT_NAMES else "other"

    except Exception:
        return "other"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PAIRS_PATH)
    print(f"Total pairs to label: {len(df):,}")

    # Resume from checkpoint if partially done
    if OUT_PATH.exists():
        done_df = pd.read_csv(OUT_PATH)
        done_ids = set(done_df["tweet_id"].astype(str).tolist())
        remaining = df[~df["tweet_id"].astype(str).isin(done_ids)]
        print(f"Resuming — {len(done_ids):,} already labeled, {len(remaining):,} remaining")
        results = done_df.to_dict("records")
    else:
        remaining = df
        results = []

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Labeling"):
        message = str(row["customer_clean"])
        intent  = classify_one(message)

        record = row.to_dict()
        record["intent"] = intent
        results.append(record)

        # Save checkpoint every 500 rows
        if len(results) % 500 == 0:
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)

        time.sleep(SLEEP_SEC)

    # Final save
    labeled_df = pd.DataFrame(results)
    labeled_df.to_csv(OUT_PATH, index=False)
    print(f"\nDone! Saved {len(labeled_df):,} labeled pairs → {OUT_PATH}")

    # Print distribution
    print("\nIntent distribution:")
    print(labeled_df["intent"].value_counts())


if __name__ == "__main__":
    main()
