"""
LangGraph Agent Pipeline — Thread-Aware Flow
=============================================
Flow:
  Customer Message + Thread Context
        ↓
  Classify Intent (uses thread history to resolve vague references)
        ↓
  Route:
    ├── Single clear intent, high/med confidence, no rubric rule fired → RAG (intent-filtered) → Draft reply → Send
    └── Multi-intent / low confidence / rubric rule fired → Escalate to human

Key improvement: Thread context resolves pronouns like "this item",
"my order", "that package" by looking at previous messages in the
conversation thread.

DECISION (see claude.md decision log): the rule-based part of escalation
uses agent.escalation._check_rules() -- the SAME rubric golden_set.csv
was hand-labeled against (PII placeholders, security language, disputed
charges, staff misconduct, repeat contact, account lockout, too-vague-to-
act-on) -- instead of a separate ad hoc keyword list. Multi-intent and
low-confidence are kept as ADDITIONAL escalation triggers on top of that
rubric, since they're real production concerns the golden set (sampled as
single messages) never tested. This keeps Step 6's pipeline-vs-baseline
escalation comparison apples-to-apples instead of comparing two different
escalation policies.

Models:
- openai/gpt-oss-120b : structured JSON (classify)
- groq/compound       : natural text (draft reply)
"""

import json
import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

from agent.intents import INTENTS, INTENT_NAMES
from agent.retriever import ReplyRetriever
from agent.escalation import _check_rules

load_dotenv()

client         = Groq(api_key=os.getenv("GROQ_API_KEY"))
CLASSIFY_MODEL = "openai/gpt-oss-120b"
DRAFT_MODEL    = "groq/compound"

FALLBACK_REPLY = "Sorry for the trouble! Please DM us your order details and we'll look into this right away."


# ── Agent State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    customer_message:  str
    thread_context:    Optional[str]  # previous messages in thread (readable text)
    intent:            Optional[str]
    confidence:        Optional[str]
    all_intents:       Optional[list]
    retrieved_replies: Optional[list]
    draft_reply:       Optional[str]
    escalate:          Optional[bool]
    escalate_reason:   Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────
def call_groq(system: str, user: str, max_tokens: int = 300, model: str = None) -> str:
    use_model = model or CLASSIFY_MODEL
    response  = client.chat.completions.create(
        model=use_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    choice  = response.choices[0]
    content = choice.message.content
    if not content:
        content = getattr(choice.message, "reasoning_content", "") or ""
    return content.strip() if content else ""


def extract_json(raw: str) -> dict:
    raw   = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(raw)


def clean_draft(text: str) -> str:
    """Strip groq/compound reasoning steps before the actual reply."""
    if not text:
        return ""
    if "**" in text:
        parts        = text.split("**")
        header_words = ("reasoning", "final", "reply", "tweet", "response", "note", "answer")
        candidates   = [
            p.strip() for p in parts
            if p.strip() and not p.strip().lower().startswith(header_words)
        ]
        if candidates:
            paragraphs = [p.strip() for p in candidates[-1].split("\n") if p.strip()]
            if paragraphs:
                return paragraphs[-1]
    return text.strip()


def clean_historical(reply: str) -> str:
    """Remove trailing URL artifacts from historical brand replies."""
    reply = re.sub(r'here:\s*\(\d+\)', '', reply)
    reply = re.sub(r'here:\s*$',       '', reply.strip())
    reply = re.sub(r'link:\s*$',       '', reply.strip())
    reply = re.sub(r':\s*$',           '', reply.strip())
    return reply.strip()


# ── Node 1: Intent Classifier ──────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    """
    Classifies using BOTH the current message AND thread context.

    Thread context (previous 2-3 messages) resolves vague references:
      "I want to return this item" — 'this item' is resolved from:
        Previous: "My Kindle Fire stopped working"
      → classified as return_refund with HIGH confidence instead of 'other (low)'

    Detects ALL intents present — multi-intent triggers escalation.
    """
    intent_lines = "\n".join(
        f"{i+1}. {intent['name']}: {intent['description'][:90]}"
        for i, intent in enumerate(INTENTS)
    )
    examples = "\n".join(
        f'  "{i["examples"][0][:80]}" -> {i["name"]}\n'
        f'  "{i["examples"][1][:80]}" -> {i["name"]}'
        for i in INTENTS
    )

    # Thread context section
    thread_context = (state.get("thread_context") or "").strip()
    context_section = ""
    if thread_context:
        context_section = (
            f"Conversation history (most recent last):\n"
            f"{thread_context}\n\n"
            "Use conversation history to resolve any vague references "
            "('this item', 'my order', 'that package') in the current message.\n\n"
        )

    system = (
        "You are an intent classifier for Amazon customer support tweets.\n\n"
        "Intents:\n"
        f"{intent_lines}\n\n"
        "Examples:\n"
        f"{examples}\n\n"
        f"{context_section}"
        "Rules:\n"
        "- Identify ALL intents in the current message (can be more than one).\n"
        "- primary_intent = the most urgent one.\n"
        "- Use conversation history to understand context when needed.\n"
        "- Use 'other' only if truly nothing fits.\n"
        "- confidence: high=certain, medium=likely, low=unsure\n\n"
        'Output JSON only:\n'
        '{"primary_intent": "<name>", "all_intents": ["<name>"], "confidence": "high|medium|low"}'
    )

    user = f'Current message: "{state["customer_message"]}"'
    raw  = call_groq(system, user, max_tokens=150, model=CLASSIFY_MODEL)

    try:
        parsed      = extract_json(raw)
        intent      = parsed.get("primary_intent", "other")
        all_intents = parsed.get("all_intents", [intent])
        confidence  = parsed.get("confidence", "medium")
    except (json.JSONDecodeError, KeyError, ValueError):
        intent      = "other"
        all_intents = ["other"]
        confidence  = "low"

    intent      = intent if intent in INTENT_NAMES else "other"
    all_intents = [i for i in all_intents if i in INTENT_NAMES] or ["other"]

    return {**state, "intent": intent, "all_intents": all_intents, "confidence": confidence}


# ── Router ─────────────────────────────────────────────────────────────────
def route_after_classify(state: AgentState) -> str:
    message     = state["customer_message"]
    intent      = state["intent"]
    all_intents = state.get("all_intents", [intent])
    confidence  = state["confidence"]

    if len(all_intents) > 1:
        return "escalate"

    if confidence == "low":
        return "escalate"

    if _check_rules(message, intent) is not None:
        return "escalate"

    return "draft"


# ── Node 2: Reply Drafter ──────────────────────────────────────────────────
def draft_reply(state: AgentState, retriever: ReplyRetriever) -> AgentState:
    """
    Intent-filtered RAG — retrieves historical replies from SAME intent bucket.
    Falls back to unfiltered if filtered returns nothing.
    Includes thread context so reply is coherent with conversation history.
    """
    message        = state["customer_message"]
    intent         = state["intent"]
    thread_context = (state.get("thread_context") or "").strip()

    similar_replies = retriever.retrieve(message, intent=intent, top_k=3)
    trimmed         = [clean_historical(r)[:150] for r in similar_replies if clean_historical(r)]
    historical      = "\n".join(f"- {r}" for r in trimmed) if trimmed else ""

    hist_section = (
        f"How Amazon replied to similar {intent} issues (use for TONE only):\n{historical}\n\n"
        if historical else ""
    )

    context_section = (
        f"Conversation history so far:\n{thread_context}\n\n"
        if thread_context else ""
    )

    system = (
        "You are an Amazon customer support agent replying on Twitter.\n\n"
        f"Customer issue type: {intent}\n\n"
        f"{context_section}"
        f"{hist_section}"
        "Reply style:\n"
        "- Start with empathy: 'Sorry to hear...', 'Oh no!'\n"
        "- Be specific to the customer's exact issue\n"
        "- For delivery: ask for order number via DM\n"
        "- For returns/refunds: acknowledge, guide to return process\n"
        "- For account: guide to secure account, suggest phone/chat\n"
        "- For billing: acknowledge, ask for order number via DM\n"
        "- For device: give one practical troubleshooting step first\n"
        "- Under 280 characters. Do not start with 'I'.\n"
        "- Output reply text ONLY. No markdown, no reasoning."
    )

    user  = f'Current customer tweet: "{message}"'
    raw   = call_groq(system, user, max_tokens=300, model=DRAFT_MODEL)
    draft = clean_draft(raw)

    if not draft:
        draft = FALLBACK_REPLY
    if len(draft) > 280:
        draft = draft[:277] + "..."

    return {
        **state,
        "retrieved_replies": similar_replies,
        "draft_reply":       draft,
        "escalate":          False,
        "escalate_reason":   "",
    }


# ── Node 3: Escalation Handler ─────────────────────────────────────────────
def handle_escalation(state: AgentState) -> AgentState:
    """
    Mirrors route_after_classify's checks, in the same priority order, so
    the reason given always matches the reason that actually triggered
    escalation. Does NOT call the LLM-fallback branch of
    agent.escalation.decide_escalation() (only its rule-based _check_rules)
    -- that fallback is reserved for genuinely ambiguous messages with no
    rule hit, which route_after_classify never routes here anyway (those
    go to "draft"). Skipping it here avoids a second LLM call on every
    escalated message purely to re-derive a reason we can already state
    deterministically.
    """
    message     = state["customer_message"]
    intent      = state["intent"]
    all_intents = state.get("all_intents", [intent])
    confidence  = state["confidence"]

    if len(all_intents) > 1:
        reason = f"Multiple intents detected: {', '.join(all_intents)}. Complex issue — human agent needed."
    elif confidence == "low":
        reason = "Intent unclear (low classifier confidence). Escalating to avoid mishandling."
    else:
        rule_hit = _check_rules(message, intent)
        reason = rule_hit["reason"] if rule_hit else "Escalated for human review."

    return {
        **state,
        "escalate":        True,
        "escalate_reason": reason,
        "draft_reply":     None,
        "retrieved_replies": [],
    }


# ── Build Graph ────────────────────────────────────────────────────────────
def build_agent(retriever: ReplyRetriever):
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_intent)
    graph.add_node("draft",    lambda s: draft_reply(s, retriever))
    graph.add_node("escalate", handle_escalation)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"draft": "draft", "escalate": "escalate"},
    )
    graph.add_edge("draft",    END)
    graph.add_edge("escalate", END)

    return graph.compile()


def run_agent(message: str, retriever: ReplyRetriever, thread_context: str = "") -> dict:
    """
    Main entry point.
    thread_context: previous messages in thread as readable text (optional).
    """
    agent = build_agent(retriever)
    initial_state: AgentState = {
        "customer_message":  message,
        "thread_context":    thread_context,
        "intent":            None,
        "confidence":        None,
        "all_intents":       None,
        "retrieved_replies": None,
        "draft_reply":       None,
        "escalate":          None,
        "escalate_reason":   None,
    }
    result = agent.invoke(initial_state)
    return {
        "intent":          result["intent"],
        "all_intents":     result["all_intents"],
        "confidence":      result["confidence"],
        "draft_reply":     result["draft_reply"],
        "escalate":        result["escalate"],
        "escalate_reason": result["escalate_reason"],
    }
