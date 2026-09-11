"""
Shared config constants for the agent pipeline.
Kept in one place so scripts/02_build_vector_store.py and the runtime
pipeline (agent/rag.py, agent/pipeline.py) never drift out of sync.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMAZON_PAIRS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_pairs_with_context.csv")
GOLDEN_SET_CSV = os.path.join(PROJECT_ROOT, "data", "eval", "golden_set.csv")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chroma")

# ── Vector store ───────────────────────────────────────────────────────
COLLECTION_NAME = "amazon_support_pairs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
# Minimum intent-filtered candidates required before we fall back to an
# unfiltered (whole-corpus) similarity search. Protects small intents
# (e.g. a rare candidate-intent bucket) from starving retrieval.
MIN_FILTERED_CANDIDATES = TOP_K

# ── LLM ────────────────────────────────────────────────────────────────
GROQ_MODEL         = "openai/gpt-oss-120b"   # for classify + escalate (structured JSON)
GROQ_DRAFT_MODEL   = "groq/compound"           # for reply drafting (text generation)
CLASSIFIER_TEMPERATURE = 0.1
DRAFTER_TEMPERATURE = 0.3
ESCALATION_TEMPERATURE = 0.1

# ── Escalation rubric (mirrors data/eval/golden_set_methodology.md) ────
# Keep these in sync with the methodology doc by hand -- if you change
# the rubric here, update that doc too, since the golden set's escalate
# labels were built against this same rubric.
ESCALATION_KEYWORDS = {
    "security": ["hacked", "compromised", "stolen", "unauthorized access", "phishing", "phished"],
    "pii_placeholder": ["__credit_card__", "__email__", "__phone__", "__ssn__", "__address__"],
    "legal": ["lawyer", "attorney", "lawsuit", "sue you", "sue amazon", "going to sue", "legal action", "bbb complaint", "ftc"],
    "disputed_charge": ["never authorized", "don't recognize this charge", "dont recognize this charge",
                          "unauthorized charge", "fraudulent charge"],
    "staff_misconduct": ["rude", "yelled at me", "hung up on me", "unprofessional", "racist", "discriminat"],
    "repeat_contact": ["called 3 times", "called three times", "5th time", "fifth time", "again and again",
                         "still not resolved", "nobody has helped", "no one has helped", "multiple times"],
}
MIN_MESSAGE_LENGTH_FOR_AUTO_HANDLE = 20  # chars; below this, treat as too-vague -> escalate
