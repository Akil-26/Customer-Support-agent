"""
Retrieval layer for Node 2 (Reply Drafter).

DECISION (see claude.md decision log): retrieval is filtered by intent
before similarity search, not a plain intent -> canned-reply lookup and
not an unfiltered corpus-wide search.

  - Plain intent -> canned reply loses per-message specificity and is
    literally the trivial baseline (Step 5) -- it must NOT be the real
    retrieval path.
  - Unfiltered corpus search is noisier: a billing message can retrieve
    a delivery reply as an embedding-neighbor purely by coincidence.
  - Intent-filtered search keeps retrieval grounded in the single closest
    historical case *within the same problem category*, which is what
    the assignment brief asks for ("grounded in how Amazon historically
    resolved similar issues").
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

from agent.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K,
    MIN_FILTERED_CANDIDATES,
)

_client = None
_collection = None
_embed_fn = None


class _STEmbedFunction:
    """Version-safe SentenceTransformer wrapper for chromadb."""
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._model.encode(input, convert_to_numpy=True).tolist()


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = _STEmbedFunction(EMBEDDING_MODEL)
    return _embed_fn


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def get_or_create_collection():
    """
    Returns the persistent collection. Does NOT populate it -- population
    happens once via scripts/02_build_vector_store.py. If the collection
    is empty, callers should run that script first.
    """
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def collection_is_empty() -> bool:
    return get_or_create_collection().count() == 0


def retrieve_similar(customer_message: str, intent: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top_k most similar historical (customer_message, brand_reply)
    pairs, filtered to the same intent bucket first.

    Falls back to an unfiltered search if the intent-filtered pool doesn't
    have enough candidates (protects small/rare intent buckets from
    returning too few or zero examples).

    Returns: [{"customer_message": str, "brand_reply": str, "tweet_id": str,
               "intent": str, "distance": float}, ...]
    """
    collection = get_or_create_collection()

    def _run_query(where_filter):
        result = collection.query(
            query_texts=[customer_message],
            n_results=top_k,
            where=where_filter,
        )
        out = []
        if not result["ids"] or not result["ids"][0]:
            return out
        for i in range(len(result["ids"][0])):
            meta = result["metadatas"][0][i]
            out.append({
                "customer_message": result["documents"][0][i],
                "brand_reply": meta.get("brand_reply", ""),
                "tweet_id": meta.get("tweet_id", ""),
                "intent": meta.get("intent", ""),
                "distance": result["distances"][0][i],
            })
        return out

    filtered = _run_query({"intent": intent})

    if len(filtered) >= min(top_k, MIN_FILTERED_CANDIDATES):
        return filtered

    # Fallback: not enough same-intent examples -- widen to whole corpus,
    # but keep the intent-filtered hits first (they're still the best match).
    unfiltered = _run_query(None)
    seen_ids = {ex["tweet_id"] for ex in filtered}
    for ex in unfiltered:
        if len(filtered) >= top_k:
            break
        if ex["tweet_id"] not in seen_ids:
            filtered.append(ex)
            seen_ids.add(ex["tweet_id"])

    return filtered[:top_k]
