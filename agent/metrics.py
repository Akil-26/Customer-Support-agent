"""
Step 6 -- Reply-quality metrics (credential-free layer).
=========================================================
Two automatic proxies for "is this reply any good", both computed WITHOUT
any LLM call (so this module stays inside the credential-free reproduction
path, unlike agent/judge.py which will need one):

  1. ROUGE-L  -- n-gram/LCS overlap with a reference reply. Cheap, but
     penalizes any good reply that paraphrases instead of copying wording.
  2. Embedding cosine similarity (all-MiniLM-L6-v2, the SAME model already
     used by agent/retriever.py for RAG, via agent.config.EMBEDDING_MODEL)
     -- captures semantic closeness even when wording differs completely.

Why lead with embedding similarity, not ROUGE-L: see
find_rouge_failure_examples() below. It surfaces a REAL case from OUR data
(not invented, not copied from GroundReply's report -- see claude.md Gap
Analysis) where a topically-correct, differently-worded reply scores low
on ROUGE-L despite being a strong semantic match. Run it once
scripts/05_run_pipeline_on_golden.py has produced real pipeline_reply
values, then drop the actual output into the report's metrics section.

What this module does NOT cover (see claude.md Step 6 plan):
  - Groundedness / helpfulness / tone / safety -- needs an LLM judge
    (agent/judge.py, not yet built). No n-gram or embedding metric can
    check "did this reply correctly reference the customer's actual
    order status" or "is this safe to auto-send".
  - Degradation-validation suite and human-agreement kappa -- separate
    scripts; they need deliberately-corrupted variants or real human
    labels, not just a scoring function.

Reference reply used throughout = golden_set.csv's own brand_reply_clean:
the real historical Amazon reply to THAT SPECIFIC customer message. This
is a reasonable proxy for "did we land near a resolution Amazon actually
used", not a hand-curated "ideal reply" -- carry that caveat into the
report the same way the Step 5 section already caveats the simple
baseline's intent numbers.
"""
from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agent.config import EMBEDDING_MODEL

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_embedder = None  # lazy singleton, see _get_embedder()


def _clean_text(v) -> str:
    """Safe string coercion for values read back from a CSV. pandas
    represents a missing cell (e.g. pipeline_reply on an escalated row,
    which was written as Python None) as float('nan') on reload, not None
    or "" -- and NaN is truthy, so the old '(v or "").strip()' pattern
    called .strip() on a float and crashed. Handle None/NaN/non-str
    explicitly instead of relying on truthiness."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Classification metrics (thin, reusable wrappers) ─────────────────────
# 04_baselines.py and 05_run_pipeline_on_golden.py each compute these
# inline with sklearn directly; kept here too so any NEW script (e.g. a
# combined report-metrics builder) has one place to call instead of a
# third copy of the same four lines.
def intent_accuracy(gold_intents, pred_intents) -> float:
    return float(accuracy_score(gold_intents, pred_intents))


def escalation_prf(gold_escalate, pred_escalate) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        gold_escalate, pred_escalate, average="binary", zero_division=0
    )
    acc = accuracy_score(gold_escalate, pred_escalate)
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }


# ── Reply-quality metrics ─────────────────────────────────────────────────
def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F-measure, 0-1. Empty strings score 0.0, not an error --
    an escalated row with no draft_reply is a legitimate 'no reply to
    score' case, not a crash."""
    reference = _clean_text(reference)
    hypothesis = _clean_text(hypothesis)
    if not reference or not hypothesis:
        return 0.0
    return float(_rouge.score(reference, hypothesis)["rougeL"].fmeasure)


def embedding_similarity(reference: str, hypothesis: str) -> float:
    """Cosine similarity of sentence embeddings, clipped to 0-1."""
    reference = _clean_text(reference)
    hypothesis = _clean_text(hypothesis)
    if not reference or not hypothesis:
        return 0.0
    model = _get_embedder()
    vecs = model.encode([reference, hypothesis], convert_to_numpy=True, show_progress_bar=False)
    a, b = vecs[0], vecs[1]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return max(0.0, min(1.0, cos))


def batch_reply_quality(references: list[str], hypotheses: list[str]) -> list[dict]:
    """Per-row rouge_l + embedding_similarity. Embeddings are computed in
    one batched encode() call (not one-by-one) since this runs over the
    full 200-row golden set for up to 3 systems at once."""
    refs = [_clean_text(r) for r in references]
    hyps = [_clean_text(h) for h in hypotheses]

    rouge_scores = [rouge_l(r, h) for r, h in zip(refs, hyps)]

    embed_scores = [0.0] * len(refs)
    valid_idx = [i for i, (r, h) in enumerate(zip(refs, hyps)) if r and h]
    if valid_idx:
        model = _get_embedder()
        all_texts = [refs[i] for i in valid_idx] + [hyps[i] for i in valid_idx]
        vecs = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
        n = len(valid_idx)
        for k, i in enumerate(valid_idx):
            a, b = vecs[k], vecs[n + k]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            embed_scores[i] = max(0.0, min(1.0, cos))

    return [
        {"rouge_l": round(rs, 4), "embedding_similarity": round(es, 4)}
        for rs, es in zip(rouge_scores, embed_scores)
    ]


# ── Composite reply-quality score ─────────────────────────────────────────
# Weights lead with embedding similarity, not ROUGE-L, precisely because of
# the paraphrase problem this module documents (find_rouge_failure_examples
# below). ROUGE-L still keeps nonzero weight: for a SUPPORT bot specifically,
# near-verbatim reuse of a known-good historical reply is a genuine, if
# mild, positive signal -- it means the draft hews close to a resolution
# Amazon actually used, which is lower-risk than a fully free-form
# paraphrase even when both score similarly on embeddings. That's a
# support-domain-specific argument, independently derived for AmazonHelp,
# not copied from GroundReply's differently-weighted reply-suggestion
# setting -- see claude.md decision log.
REPLY_QUALITY_WEIGHTS = {"embedding_similarity": 0.7, "rouge_l": 0.3}


def composite_reply_score(reference: str, hypothesis: str) -> dict:
    rl = rouge_l(reference, hypothesis)
    es = embedding_similarity(reference, hypothesis)
    score = (
        REPLY_QUALITY_WEIGHTS["embedding_similarity"] * es
        + REPLY_QUALITY_WEIGHTS["rouge_l"] * rl
    )
    return {"rouge_l": round(rl, 4), "embedding_similarity": round(es, 4), "composite": round(score, 4)}


# ── Worked example: why ROUGE-L alone is misleading ───────────────────────
def find_rouge_failure_examples(references: list[str], hypotheses: list[str], top_n: int = 3) -> list[dict]:
    """
    Finds real rows -- from data the CALLER supplies, this function
    fabricates nothing -- where embedding similarity is high but ROUGE-L
    is low: a reply that paraphrases a good resolution rather than
    copying its wording, and gets penalized for it. Ranked by
    (embedding_similarity - rouge_l), descending.

    Intended use: pass (golden brand_reply_clean, pipeline_reply) once
    scripts/05_run_pipeline_on_golden.py has produced real predictions,
    then use the top result as the report's mandatory "why ROUGE/BLEU
    alone is misleading" worked example.
    """
    rows = batch_reply_quality(references, hypotheses)
    scored = [
        {
            "reference": references[i],
            "hypothesis": hypotheses[i],
            "rouge_l": rows[i]["rouge_l"],
            "embedding_similarity": rows[i]["embedding_similarity"],
            "gap": round(rows[i]["embedding_similarity"] - rows[i]["rouge_l"], 4),
        }
        for i in range(len(references))
        if references[i] and hypotheses[i]
    ]
    scored.sort(key=lambda r: r["gap"], reverse=True)
    return scored[:top_n]
