"""
Node 1 -- Intent Classifier.

Uses the Groq LLM + the few-shot examples already curated in
agent/intents.py (get_intent_list_block, get_few_shot_block) to classify
an incoming customer message into one of the 8 intents.

This is deliberately kept separate from agent/intents.py's
candidate_intent_by_keywords(), which is a cheap heuristic used ONLY to
tag the RAG corpus for retrieval filtering (see agent/rag.py docstring).
This module is the real, evaluated classifier (Node 1 in claude.md's
planned architecture) -- its output is what gets compared against
golden_set.csv's hand-labeled `intent` column in Step 6.
"""
from agent.llm_client import chat_json
from agent.config import CLASSIFIER_TEMPERATURE
from agent.intents import INTENT_NAMES, get_intent_list_block, get_few_shot_block

_SYSTEM_PROMPT = f"""You are an intent classifier for AmazonHelp customer support messages
posted on Twitter. Classify the customer's message into EXACTLY ONE of these intents:

{get_intent_list_block()}

Rules:
- Pick the intent that best matches what the customer is asking for or complaining about.
- If a message could fit multiple intents, pick the one matching the customer's PRIMARY ask.
- If nothing fits clearly, use "other". Do not force a bad fit.
- Respond with ONLY a JSON object: {{"intent": "<intent_name>", "confidence": <0.0-1.0>}}
- "confidence" reflects how clearly the message matches that intent, not how important it is.

Examples:
{get_few_shot_block()}
"""


def classify_intent(customer_message: str) -> dict:
    """
    Returns {"intent": str, "confidence": float}.
    Falls back to {"intent": "other", "confidence": 0.0} on any LLM/parse
    failure -- a classification node must never crash the pipeline.
    """
    result = chat_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f'Message: "{customer_message}"',
        temperature=CLASSIFIER_TEMPERATURE,
        max_tokens=100,
    )

    if result.get("_parse_error"):
        return {"intent": "other", "confidence": 0.0, "_error": "llm_parse_failure"}

    intent = result.get("intent", "other")
    if intent not in INTENT_NAMES:
        intent = "other"

    try:
        confidence = float(result.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    return {"intent": intent, "confidence": confidence}
