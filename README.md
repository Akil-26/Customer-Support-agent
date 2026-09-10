# Amazon Support AI Agent
**Hiver SDE Intern Take-Home Assignment**

An AI customer support agent for **AmazonHelp** (Twitter) built with LangGraph + Claude API.

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
| LLM | Claude API (claude-sonnet-4-6) |
| Vector store | ChromaDB (local) |
| Embeddings | all-MiniLM-L6-v2 |
| API serving | FastAPI |

---

## Project Structure

```
Amazon-support-agent/
├── agent/
│   └── intents.py          ← 8 intent definitions + prompt builders
├── scripts/
│   └── 01_explore_and_clean.py  ← Data cleaning pipeline
├── data/
│   ├── raw/                ← Raw Kaggle dataset (not in repo)
│   └── processed/          ← Cleaned outputs (not in repo)
├── claude.md               ← Full project context and status
└── README.md
```

---

## Setup

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
# Add your ANTHROPIC_API_KEY in .env

# 5. Download dataset
# Get twcs.csv from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# Place at: data/raw/twcs/twcs.csv

# 6. Run data cleaning
python scripts/01_explore_and_clean.py
```

---

## Status

- [x] Step 1 — Data Exploration & Cleaning
- [x] Step 2 — Intent Definition
- [ ] Step 3 — Golden Evaluation Set
- [ ] Step 4 — Agent Pipeline
- [ ] Step 5 — Baselines
- [ ] Step 6 — Evaluation Harness

---

## Dataset

**Primary:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 2.8M tweets  
**Brand:** AmazonHelp — 40,861 clean (customer, reply) pairs after filtering
