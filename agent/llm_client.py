"""
Groq LLM client wrapper.
Centralizes model name, temperature defaults, and JSON-parsing so every
node in the pipeline calls the LLM the same way and fails the same way.
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv

from agent.config import GROQ_MODEL

load_dotenv()

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2,
         max_tokens: int = 500, json_mode: bool = False) -> str:
    """Single-turn chat completion. Returns raw text content."""
    client = get_client()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.1,
              max_tokens: int = 500) -> dict:
    """
    Chat completion forced into JSON mode, parsed into a dict.
    Falls back to a safe error dict on parse failure rather than crashing
    the whole pipeline on one bad LLM response -- callers must check for
    the "_parse_error" key and handle it (fail-safe default, not a crash).
    """
    raw = chat(system_prompt, user_prompt, temperature, max_tokens, json_mode=True)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"_parse_error": True, "_raw": raw}
