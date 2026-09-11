"""
Node 3 -- Escalation Decider.

Rules first, LLM only for genuinely ambiguous leftovers -- this mirrors
how golden_set.csv's `escalate` column was actually labeled (see
golden_set_methodology.md: "implemented as rules, then spot-checked by
hand"). Using the same rubric here means Step 6 evaluation is comparing
the pipeline against a rubric it was actually designed to match, not a
different standard.

Rubric (from data/eval/golden_set_methodology.md), escalate = True when:
  - Redacted account/payment identifiers needing secure manual verification
  - Account security language (hacked, stolen, compromised, phishing)
  - Legal/regulatory escalation language
  - An unrecognized or disputed charge needing investigation
  - Staff misconduct reports
  - Evidence of repeat failed contact
  - Account lockout/suspension (needs identity verification, not a script)
  - Message too short/vague to identify the actual issue
  Otherwise: auto-handle.
"""
from agent.llm_client import chat_json
from agent.config import (
    ESCALATION_KEYWORDS, ESCALATION_TEMPERATURE, MIN_MESSAGE_LENGTH_FOR_AUTO_HANDLE,
)

_SYSTEM_PROMPT = """You decide whether an AmazonHelp customer support message needs to be
escalated to a human agent, using this rubric. Escalate (true) if the message shows:
- Account security issues (hacked, compromised, stolen access)
- Legal or regulatory threats
- A disputed/unrecognized charge needing investigation
- Staff misconduct reports
- The customer has already tried multiple times without resolution
- The message is too vague/unclear to act on safely
Otherwise, escalate should be false (safe to auto-handle with a standard reply).

Respond with ONLY JSON: {"escalate": true/false, "reason": "<short reason, under 15 words>"}
"""


def _check_rules(customer_message: str, intent: str) -> dict | None:
    """Returns a decision dict if a rule fires, else None (ambiguous -> LLM)."""
    text_lower = customer_message.lower()

    if len(customer_message.strip()) < MIN_MESSAGE_LENGTH_FOR_AUTO_HANDLE:
        return {"escalate": True, "reason": "Message too short/vague to act on safely."}

    for category, keywords in ESCALATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                readable = category.replace("_", " ")
                return {"escalate": True, "reason": f"Message matches '{readable}' escalation rule."}

    # Account lockout: per golden set, account_access has the highest escalation
    # rate (38%) because login/lockout issues usually need identity verification.
    lockout_terms = ["locked", "cant log in", "can't log in", "cannot log in",
                      "suspended", "cant access", "can't access", "account suspended",
                      "account locked", "locked out"]
    if intent == "account_access" and any(term in text_lower for term in lockout_terms):
        return {"escalate": True, "reason": "Account lockout needs identity verification."}

    return None  # no rule fired -- ambiguous, defer to LLM


def decide_escalation(customer_message: str, intent: str) -> dict:
    """
    Returns {"escalate": bool, "escalate_reason": str}.
    """
    rule_hit = _check_rules(customer_message, intent)
    if rule_hit is not None:
        return {"escalate": rule_hit["escalate"], "escalate_reason": rule_hit["reason"]}

    # No rule fired -- ambiguous case, ask the LLM (mirrors claude.md's
    # planned "Rules first -> Claude for ambiguous" design).
    result = chat_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f'Message: "{customer_message}"\nIntent: {intent}',
        temperature=ESCALATION_TEMPERATURE,
        max_tokens=80,
    )

    if result.get("_parse_error"):
        # Fail safe: if we can't parse the LLM's decision, escalate rather
        # than silently auto-handling something we couldn't evaluate.
        return {"escalate": True, "escalate_reason": "Escalation check failed -- routed to human as a precaution."}

    escalate = bool(result.get("escalate", True))
    reason = result.get("reason", "No standard resolution pattern found for this message.")
    return {"escalate": escalate, "escalate_reason": reason}
