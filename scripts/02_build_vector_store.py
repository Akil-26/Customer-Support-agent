"""
Step 4 prep — Build the ChromaDB vector store for RAG retrieval.

Reads data/processed/amazon_pairs.csv, keeps only rows WITH a brand reply
(has_reply == True -- 40,324 of 40,861 rows, per claude.md Step 1 output),
tags each row with a candidate_intent via keyword scoring (agent/intents.py
candidate_intent_by_keywords -- the same logic used for golden_set.csv),
embeds the customer message with all-MiniLM-L6-v2, and persists everything
to data/chroma/ so agent/rag.py can do intent-filtered similarity search
at runtime without re-embedding.

Run once (or whenever amazon_pairs.csv changes):
    python scripts/02_build_vector_store.py

Expected runtime: a few minutes on CPU for ~40k short tweets.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from agent.config import AMAZON_PAIRS_CSV, COLLECTION_NAME
from agent.intents import candidate_intent_by_keywords
from agent.rag import get_client, get_or_create_collection

BATCH_SIZE = 512


def main():
    print(f"Loading {AMAZON_PAIRS_CSV} ...")
    df = pd.read_csv(AMAZON_PAIRS_CSV)
    before = len(df)

    df = df[df["has_reply"] == True].copy()
    df = df.dropna(subset=["customer_clean", "brand_reply_clean"])
    df = df[df["customer_clean"].str.strip() != ""]
    print(f"Loaded {before} rows -> {len(df)} rows with a usable brand reply")

    print("Scoring candidate intents via keyword matching ...")
    tqdm.pandas(desc="Tagging intents")
    df["candidate_intent"] = df["customer_clean"].progress_apply(candidate_intent_by_keywords)

    print("\nCandidate intent distribution (heuristic, for RAG filtering only):")
    print(df["candidate_intent"].value_counts())

    client = get_client()
    # Fresh build every run -- avoids duplicate/stale rows if amazon_pairs.csv changed.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"\nDropped existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = get_or_create_collection()

    ids = df["tweet_id"].astype(str).tolist()
    documents = df["customer_clean"].tolist()
    metadatas = [
        {
            "tweet_id": str(row.tweet_id),
            "intent": row.candidate_intent,
            "brand_reply": row.brand_reply_clean,
        }
        for row in df.itertuples()
    ]

    print(f"\nEmbedding + inserting {len(ids)} rows in batches of {BATCH_SIZE} ...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Building vector store"):
        collection.add(
            ids=ids[i:i + BATCH_SIZE],
            documents=documents[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} rows.")
    print("data/chroma/ is persistent -- no need to rebuild unless amazon_pairs.csv changes.")


if __name__ == "__main__":
    main()
