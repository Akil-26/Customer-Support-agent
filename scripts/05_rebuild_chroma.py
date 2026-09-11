"""
Script: Rebuild ChromaDB with thread-aware pairs
=================================================
Deletes old ChromaDB and rebuilds using amazon_pairs_with_context.csv
which has proper tweet_ids and thread context.

Usage:
  python scripts/05_rebuild_chroma.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.retriever import ReplyRetriever, reset_chroma_db

PAIRS_PATH = "data/processed/amazon_pairs_with_context.csv"

def main():
    print("Deleting old ChromaDB...")
    reset_chroma_db()

    print("Rebuilding with thread-aware pairs...")
    retriever = ReplyRetriever()
    retriever.build(PAIRS_PATH)
    print(f"Done! Total indexed: {retriever.collection.count():,}")

if __name__ == "__main__":
    main()
