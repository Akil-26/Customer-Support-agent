"""
Debug script — check what Groq model actually returns
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Test with simple prompt
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=100,
    messages=[
        {"role": "system", "content": "Reply with JSON only: {\"intent\": \"delivery_issue\", \"confidence\": \"high\"}"},
        {"role": "user",   "content": "my order hasn't arrived"},
    ],
    temperature=0.2,
)

choice = response.choices[0]
print("=== FULL RESPONSE ===")
print(f"content          : '{choice.message.content}'")
print(f"reasoning_content: '{getattr(choice.message, 'reasoning_content', 'N/A')}'")
print(f"finish_reason    : '{choice.finish_reason}'")
print(f"model            : '{response.model}'")
print()

# List available models
models = client.models.list()
print("=== AVAILABLE MODELS ===")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")
