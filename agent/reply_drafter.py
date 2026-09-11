"""
Node 2 -- Reply Drafter.

Retrieves the top-K historically similar (customer_message, brand_reply)
pairs -- filtered to the classified intent first, see agent/rag.py -- and
asks the LLM to draft a fresh reply grounded in that retrieved pattern.

The retrieved examples are returned alongside the draft so they can be
logged/inspected and used for the "grounded" score in the LLM-as-judge
rubric (Step 6) -- we want to be able to show WHICH historical replies
a given draft was grounded in, not just claim it was.
"""
from agent.llm_client import chat
from agent.config import DRAFTER_TEMPERATURE, TOP_K
from agent.rag import retrieve_similar

_SYSTEM_PROMPT = """You are drafting a customer support reply for AmazonHelp on Twitter.

You will be given the customer's message and 1-3 examples of how AmazonHelp
historically replied to similar customer messages. Use those examples to
understand the TONE and the TYPICAL RESOLUTION PATTERN AmazonHelp uses for
this kind of issue -- do not copy them verbatim, and do not invent specific
facts (order numbers, dates, refund amounts, names) that aren't in the
customer's message.

Rules:
- Keep it under 280 characters (Twitter reply length).
- Be empathetic but concise, matching AmazonHelp's typical tone.
- If the historical pattern is "ask the customer to DM us" or "ask for order details",
  follow that same pattern -- don't promise a resolution you can't actually deliver.
- Never invent or confirm specific account details, refund amounts, or order status
  that weren't given to you.
- Output ONLY the reply text. No preamble, no quotes, no explanation.
"""


def draft_reply(customer_message: str, intent: str, top_k: int = TOP_K) -> dict:
    """
    Returns {"draft_reply": str, "retrieved_examples": list[dict]}.
    """
    examples = retrieve_similar(customer_message, intent, top_k=top_k)

    if examples:
        examples_block = "\n\n".join(
            f'Similar past message: "{ex["customer_message"]}"\n'
            f'AmazonHelp replied: "{ex["brand_reply"]}"'
            for ex in examples
        )
    else:
        examples_block = "(No similar historical examples found -- draft from general AmazonHelp tone/policy.)"

    user_prompt = (
        f"Customer's message: \"{customer_message}\"\n"
        f"Classified intent: {intent}\n\n"
        f"Historical examples:\n{examples_block}\n\n"
        f"Draft the reply now."
    )

    draft = chat(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=DRAFTER_TEMPERATURE,
        max_tokens=150,
    )

    return {
        "draft_reply": draft.strip().strip('"'),
        "retrieved_examples": examples,
    }
