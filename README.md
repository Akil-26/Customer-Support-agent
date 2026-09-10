# Amazon Support AI Agent
**Hiver SDE Intern Take-Home Assignment**

An AI customer support agent for **AmazonHelp** (Twitter) built with LangGraph + Groq API.

---

## What it does

1. **Classifies** incoming customer messages into 8 intents
2. **Drafts** a reply grounded in how Amazon historically resolved similar issues (RAG)
3. **Decides** whether to auto-handle or escalate to a human — with a reason

---

## Intents

| Intent | Description |
|--------|-------------|
| `delivery_issue` | Package not arrived, delayed, wrong status |
| `return_refund` | Return request, refund, replacement |
| `account_access` | Login issues, locked, hacked accounts |
| `billing_payment` | Unexpected charges, cashback missing |
| `product_issue` | Damaged, wrong item, tampered package |
| `prime_subscription` | Prime benefits, Prime Video, cancel Prime |
| `device_tech_support` | Kindle, Echo, Alexa, Fire Tablet issues |
| `other` | Vague, feedback, unclear intent |

---

## Stack

| Component | Tool |
|-----------|------|
| Agent pipeline | LangGraph |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Vector store | ChromaDB (local) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| API serving | FastAPI |

---

## Project Structure

```
Amazon-support-agent/
├── agent/
│   └── intents.py               ← 8 intent definitions + prompt builders
├── scripts/
│   └── 01_explore_and_clean.py  ← Data cleaning pipeline
├── data/
│   ├── raw/                     ← Raw Kaggle dataset (not in repo — too large)
│   └── processed/               ← Cleaned outputs (not in repo)
├── .env.example                 ← API key template
├── requirements.txt             ← Python dependencies
└── README.md
```

---

## Setup & Reproduce Results

```bash
# 1. Clone
git clone https://github.com/Akil-26/Customer-Support-agent.git
cd Customer-Support-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Open .env and add your GROQ_API_KEY

# 5. Download dataset
# Get twcs.csv from Kaggle:
# https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# Place at: data/raw/twcs/twcs.csv

# 6. Run data cleaning (takes ~5 mins)
python scripts/01_explore_and_clean.py
# Output: data/processed/amazon_pairs.csv (40,861 pairs)
#         data/processed/amazon_customers.csv
#         data/processed/exploration_report.txt
```

---

## Data Pipeline

```
twcs.csv (2.8M tweets, 516MB)
        ↓
Filter AmazonHelp brand rows (169,840)
        ↓
Collect customer messages AmazonHelp replied to (154,976)
        ↓
Clean: remove @mentions, URLs, emojis, agent codes, non-English
        ↓
Filter: English only, min 20 chars, min 4 words, no mid-conversation
        ↓
Deduplicate on cleaned text
        ↓
40,861 clean (customer, brand_reply) pairs
```

---

## Progress

- [x] Step 1 — Data Exploration & Cleaning (40,861 clean pairs)
- [x] Step 2 — Intent Definition (8 intents from 500+ real messages)
- [ ] Step 3 — Golden Evaluation Set (150-250 hand-labelled examples)
- [ ] Step 4 — Agent Pipeline (LangGraph 3-node: Classify → Draft → Escalate)
- [ ] Step 5 — Baselines (Trivial + TF-IDF Simple)
- [ ] Step 6 — Evaluation Harness (metrics + LLM-as-judge + human agreement)

---

## Dataset

**Primary:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 2.8M tweets, multi-brand  
**Brand chosen:** AmazonHelp  
**After cleaning:** 40,861 English-only (customer message, brand reply) pairs  
**Date range:** April 2017 – September 2017
