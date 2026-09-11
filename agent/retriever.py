"""
RAG Retriever — ChromaDB (ACTIVE, used by the current agent/pipeline.py)
=========================
NOTE: an earlier session marked this file "superseded" in favor of
agent/rag.py + agent/classifier.py/reply_drafter.py/escalation.py. That
refactor was since abandoned in favor of THIS file + agent/pipeline.py's
run_agent(), which added thread-context and multi-intent support that the
rag.py-based refactor didn't have. agent/rag.py, classifier.py,
reply_drafter.py, llm_client.py are now the orphaned ones — not imported
by agent/pipeline.py. agent/escalation.py is NOT orphaned: pipeline.py
imports and calls its _check_rules() directly (see pipeline.py's
route_after_classify() / handle_escalation()), and scripts/04_baselines.py
uses the same _check_rules() for the simple baseline's escalation rule.
One shared rubric, used consistently everywhere — a prior version of this
note claimed two different escalation rule sets existed; that was stale
and has been corrected (verified against pipeline.py's actual imports).
=========================
Indexes all (customer_message, brand_reply) pairs from amazon_pairs.csv.
At query time, retrieves top-k most similar brand replies for a given
customer message, optionally filtered by intent.

Usage:
  retriever = ReplyRetriever()
  retriever.build("data/processed/amazon_pairs.csv")
  replies = retriever.retrieve("my order never arrived", intent="delivery_issue", top_k=3)
"""

import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION  = "amazon_replies"
EMBED_MODEL = "all-MiniLM-L6-v2"


class _STEmbedFunction:
    """
    ChromaDB 1.5+ compatible embedding function wrapper.

    ChromaDB 1.5+ requires THREE methods:
      - __call__(input)        → used internally / legacy
      - embed_documents(input) → called when ADDING documents
      - embed_query(input)     → called when QUERYING

    Without all three, ChromaDB raises AttributeError at query time
    even though indexing succeeds (that's the bug we hit before).
    """
    def __init__(self, model_name: str):
        self._model      = SentenceTransformer(model_name)
        self._model_name = model_name

    def name(self) -> str:
        return self._model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)


class ReplyRetriever:
    def __init__(self):
        self.ef     = _STEmbedFunction(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, pairs_path: str, batch_size: int = 256):
        """
        Index all pairs into ChromaDB.
        Skips if already indexed.

        We index the CUSTOMER message as the document (semantic query target)
        and store the BRAND REPLY in metadata — retrieved as output.
        This way: at query time we search by customer message similarity
        and return the matching brand reply.
        """
        if self.collection.count() > 0:
            print(f"[Retriever] Already indexed {self.collection.count():,} docs. Skipping build.")
            return

        df = pd.read_csv(pairs_path)
        df = df[df["has_reply"] == True].dropna(
            subset=["customer_clean", "brand_reply_clean"]
        ).reset_index(drop=True)

        print(f"[Retriever] Indexing {len(df):,} pairs...")

        for start in tqdm(range(0, len(df), batch_size), desc="Indexing"):
            batch = df.iloc[start: start + batch_size]
            self.collection.add(
                ids       = [str(i) for i in batch.index],
                documents = batch["customer_clean"].tolist(),
                metadatas = [
                    {
                        "brand_reply": row["brand_reply_clean"],
                        "intent":      str(row.get("intent", "unknown")),
                        "tweet_id":    str(row["tweet_id"]),
                    }
                    for _, row in batch.iterrows()
                ],
            )

        print(f"[Retriever] Done. Total indexed: {self.collection.count():,}")

    def retrieve(self, query: str, intent: str = None, top_k: int = 3) -> list[str]:
        """
        Retrieve top-k brand replies for a customer message.

        Strategy:
        1. Try intent-filtered search first (stays within same problem category)
        2. If filtered search returns nothing → fallback to full corpus search
        3. Return list of brand reply strings (empty list if nothing found)
        """
        where = {"intent": intent} if intent and intent != "other" else None

        def _query(where_filter):
            return self.collection.query(
                query_texts = [query],
                n_results   = top_k,
                where       = where_filter,
                include     = ["metadatas"],
            )

        # First attempt — with intent filter
        try:
            results = _query(where)
            replies = [
                m.get("brand_reply", "")
                for m in results["metadatas"][0]
                if m.get("brand_reply", "")
            ]
        except Exception:
            replies = []

        # Fallback — no filter if filtered returned nothing
        if not replies:
            try:
                results = _query(None)
                replies = [
                    m.get("brand_reply", "")
                    for m in results["metadatas"][0]
                    if m.get("brand_reply", "")
                ]
            except Exception:
                replies = []

        return replies


def reset_chroma_db():
    """
    Delete and rebuild ChromaDB from scratch.
    Call this when the embedding function interface changes
    (otherwise old index is incompatible with new EF).
    """
    path = Path(CHROMA_PATH)
    if path.exists():
        shutil.rmtree(path)
        print(f"[Retriever] Deleted old ChromaDB at {CHROMA_PATH}")
