"""
Debug RAG retrieval — check what ChromaDB actually returns
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.retriever import ReplyRetriever

retriever = ReplyRetriever()

test_queries = [
    "my order hasn't arrived and it's been 5 days past the delivery date",
    "I want to return this item, it arrived completely broken",
    "my Echo Show keeps disconnecting from wifi every few hours",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 60)

    # Check raw ChromaDB results
    results = retriever.collection.query(
        query_texts = [query],
        n_results   = 3,
        include     = ["metadatas", "documents", "distances"],
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"  [{i+1}] distance    : {dist:.4f}")
        print(f"       document    : {doc[:80]}")
        print(f"       brand_reply : '{meta.get('brand_reply', 'EMPTY')[:80]}'")
        print(f"       intent      : {meta.get('intent', 'MISSING')}")
        print()
