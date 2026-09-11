"""
Script: Build conversation threads for AmazonHelp
===================================================
Reconstructs full conversation threads from twcs.csv directly.
Does NOT rely on amazon_pairs.csv having tweet_id — works from raw data.

For each AmazonHelp customer message, finds the previous 2 messages
in the thread using in_response_to_tweet_id.

Output: data/processed/amazon_pairs_with_context.csv

Usage:
  python scripts/04_build_threads.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm

RAW_PATH   = Path("data/raw/twcs/twcs.csv")
OUT_PATH   = Path("data/processed/amazon_pairs_with_context.csv")
CHUNK_SIZE = 100_000
BRAND      = "AmazonHelp"


NON_ENGLISH_SIGNALS = [
    "danke", "bitte", "nicht", "und", "ist", "ich", "beim", "wird",
    "haben", "oder", "aber", "noch", "wir", "das", "der", "die",
    "merci", "bonjour", "pourquoi", "vous", "est", "une", "mon",
    "gracias", "hola", "por favor", "tengo",
    "obrigado", "nao", "voce",
]

def is_english(text: str) -> bool:
    if not isinstance(text, str) or len(text) == 0:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) >= 0.20:
        return False
    text_lower = text.lower()
    for signal in NON_ENGLISH_SIGNALS:
        if signal in text_lower:
            return False
    return True


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\^[A-Z]{2,3}", "", text)   # remove agent codes ^TN ^JS
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_full_tweet_map(raw_path: Path) -> dict:
    """Load ALL tweets keyed by tweet_id (as string)."""
    print("Loading full tweet map...")
    tweet_map = {}

    for chunk in pd.read_csv(raw_path, dtype=str, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            tid = str(row["tweet_id"]).strip()
            tweet_map[tid] = {
                "text":      str(row.get("text", "")),
                "author_id": str(row.get("author_id", "")),
                "inbound":   str(row.get("inbound", "")),
                "parent_id": str(row.get("in_response_to_tweet_id", "")).strip(),
            }

    print(f"  Loaded {len(tweet_map):,} tweets")
    return tweet_map


def get_context(tweet_id: str, tweet_map: dict, max_depth: int = 3) -> list:
    """Walk up thread from tweet_id, return previous messages oldest-first."""
    context    = []
    current_id = str(tweet_id).strip()

    for _ in range(max_depth):
        current = tweet_map.get(current_id)
        if not current:
            break

        parent_id = current["parent_id"]
        if not parent_id or parent_id in ("nan", "", "None"):
            break

        parent = tweet_map.get(parent_id)
        if not parent:
            break

        text = clean_text(parent["text"])
        if len(text) > 5:
            role = "customer" if parent["inbound"] == "True" else "amazon"
            context.insert(0, {"role": role, "text": text})

        current_id = parent_id

    return context


def context_to_text(context: list) -> str:
    if not context:
        return ""
    lines = []
    for msg in context:
        role = "Customer" if msg["role"] == "customer" else "Amazon"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load full tweet map
    tweet_map = load_full_tweet_map(RAW_PATH)

    # Rebuild AmazonHelp pairs WITH tweet_ids directly from raw data
    print("\nRe-extracting AmazonHelp pairs with tweet_ids...")
    amazon_replies = {
        tid: info for tid, info in tweet_map.items()
        if info["author_id"] == BRAND
    }
    print(f"  AmazonHelp reply tweets: {len(amazon_replies):,}")

    # For each Amazon reply, find the customer message it replied to
    rows = []
    for amazon_tid, amazon_info in tqdm(amazon_replies.items(), desc="Building pairs"):
        customer_tid = amazon_info["parent_id"]
        if not customer_tid or customer_tid in ("nan", "", "None"):
            continue

        customer = tweet_map.get(customer_tid)
        if not customer or customer["inbound"] != "True":
            continue

        customer_text = clean_text(customer["text"])
        amazon_text   = clean_text(amazon_info["text"])

        # Skip non-English pairs
        if not is_english(customer["text"]) or not is_english(amazon_info["text"]):
            continue

        if len(customer_text) < 20 or len(customer_text.split()) < 4:
            continue

        # Get thread context for this customer message
        context      = get_context(customer_tid, tweet_map, max_depth=3)
        context_text = context_to_text(context)

        rows.append({
            "tweet_id":          customer_tid,
            "customer_clean":    customer_text,
            "brand_reply_clean": amazon_text,
            "has_reply":         bool(amazon_text),
            "context_text":      context_text,
            "has_context":       len(context) > 0,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["customer_clean"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    n_ctx = df["has_context"].sum()
    print(f"\nDone! Saved {len(df):,} pairs → {OUT_PATH}")
    print(f"  With thread context    : {n_ctx:,} ({n_ctx/len(df)*100:.1f}%)")
    print(f"  Without thread context : {len(df)-n_ctx:,}")

    # Sample with context
    print("\nSample pairs WITH context:")
    sample = df[df["has_context"]].head(3)
    for _, row in sample.iterrows():
        print(f"\n  Context  : {row['context_text'].replace(chr(10), ' | ')}")
        print(f"  Message  : {row['customer_clean'][:100]}")
        print(f"  Reply    : {row['brand_reply_clean'][:80]}")


if __name__ == "__main__":
    main()
