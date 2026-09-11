"""
Step 6 (part 1) -- Run the REAL agent pipeline over data/eval/golden_set.csv
to produce predictions comparable to scripts/04_baselines.py's trivial/simple
columns.

This is the prerequisite Step 6's evaluation harness needs (see claude.md):
metrics.py, the LLM judge, and the degradation suite can't score anything
real until these predictions exist.

Design notes:
- Needs a live GROQ_API_KEY (agent.pipeline.run_agent calls Groq twice per
  auto-handled message: classify + draft; escalated messages only call
  classify, since handle_escalation() reuses the rule-based reason instead
  of a second LLM call -- see pipeline.py's docstring). This is why this
  script is deliberately kept OUT of the <15-min credential-free reproduction
  path (same reasoning as scripts/03_label_pairs.py) -- see claude.md's
  "reproducibility risk" note. Expected runtime: a few minutes for 200 rows.
- CHECKPOINTED: saves progress every CHECKPOINT_EVERY rows to a temp file
  and resumes from it on restart, so a rate-limit blip or crash on row 150
  doesn't lose rows 1-149. 400 sequential Groq calls is enough that this
  matters in practice.
- FAIL-SAFE per row: one retry on transient error, then falls back to a
  clearly-marked error record (escalate=True, reason prefixed
  "PIPELINE ERROR:") rather than crashing the whole run -- mirrors
  agent/llm_client.py's chat_json() fail-safe philosophy (escalate rather
  than silently guessing on something we couldn't evaluate).
- OUTPUT: merges into data/eval/baseline_predictions.csv (adds pipeline_*
  columns) if that file already exists (from running 04_baselines.py),
  so the result is a single file with all three systems' predictions
  side by side. If it doesn't exist yet, creates it from golden_set.csv
  alone and prints a reminder to also run 04_baselines.py for the full
  3-way comparison.

Usage:
    python scripts/05_run_pipeline_on_golden.py                # full 200 rows
    python scripts/05_run_pipeline_on_golden.py --limit 20      # quick test
    python scripts/05_run_pipeline_on_golden.py --no-resume     # ignore checkpoint
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agent.config import AMAZON_PAIRS_CSV, GOLDEN_SET_CSV
from agent.pipeline import run_agent
from agent.retriever import ReplyRetriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PREDICTIONS = os.path.join(ROOT, "data", "eval", "baseline_predictions.csv")
CHECKPOINT_PATH = os.path.join(ROOT, "data", "eval", ".pipeline_predictions_checkpoint.csv")
REPORTS_DIR = os.path.join(ROOT, "reports")

CHECKPOINT_EVERY = 20
MAX_RETRIES = 1
RETRY_SLEEP_SECONDS = 2


def _to_bool(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == 'true'
    return bool(v)


def load_base_dataframe() -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, has_baseline_columns)."""
    if os.path.exists(BASELINE_PREDICTIONS):
        print(f"Found existing {BASELINE_PREDICTIONS} -- using it as the base "
              f"(preserves trivial_*/simple_* columns for a 3-way comparison).")
        return pd.read_csv(BASELINE_PREDICTIONS), True
    print(f"No {BASELINE_PREDICTIONS} found -- loading golden_set.csv directly.")
    print("  NOTE: run scripts/04_baselines.py too if you want the full 3-way "
          "(trivial vs simple vs pipeline) comparison in one file.")
    return pd.read_csv(GOLDEN_SET_CSV), False


def load_checkpoint(n_rows: int) -> pd.DataFrame | None:
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    ckpt = pd.read_csv(CHECKPOINT_PATH)
    if len(ckpt) != n_rows or "pipeline_intent" not in ckpt.columns:
        print("  Checkpoint found but shape/columns don't match current run -- ignoring it.")
        return None
    done = ckpt["pipeline_intent"].notna().sum()
    print(f"  Resuming from checkpoint: {done}/{n_rows} rows already done.")
    return ckpt


def predict_one(message: str, retriever: ReplyRetriever, thread_context: str = "") -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = run_agent(message, retriever, thread_context=thread_context)
            all_intents = result.get("all_intents") or [result.get("intent", "other")]
            return {
                "pipeline_intent": result.get("intent", "other"),
                "pipeline_all_intents": ";".join(all_intents),
                "pipeline_confidence": result.get("confidence", "low"),
                "pipeline_reply": result.get("draft_reply"),
                "pipeline_escalate": bool(result.get("escalate", True)),
                "pipeline_escalate_reason": result.get("escalate_reason", ""),
            }
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see module docstring
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    return {
        "pipeline_intent": "other",
        "pipeline_all_intents": "other",
        "pipeline_confidence": "low",
        "pipeline_reply": None,
        "pipeline_escalate": True,
        "pipeline_escalate_reason": f"PIPELINE ERROR: {type(last_err).__name__}: {last_err}",
    }


def evaluate(golden: pd.DataFrame) -> dict:
    intent_acc = accuracy_score(golden["intent"], golden["pipeline_intent"])
    escalate_p, escalate_r, escalate_f1, _ = precision_recall_fscore_support(
        golden["escalate"], golden["pipeline_escalate"], average="binary", zero_division=0
    )
    escalate_acc = accuracy_score(golden["escalate"], golden["pipeline_escalate"])
    n_errors = golden["pipeline_escalate_reason"].astype(str).str.startswith("PIPELINE ERROR").sum()

    print(f"\n[REAL PIPELINE] vs golden_set.csv (n={len(golden)})")
    print(f"  Intent accuracy          : {intent_acc:.3f}")
    print(f"  Escalate accuracy        : {escalate_acc:.3f}")
    print(f"  Escalate precision       : {escalate_p:.3f}")
    print(f"  Escalate recall          : {escalate_r:.3f}")
    print(f"  Escalate F1              : {escalate_f1:.3f}")
    if n_errors:
        print(f"  \u26a0 rows with pipeline errors (retried, then fell back): {n_errors}")

    return {
        "intent_accuracy": round(intent_acc, 4),
        "escalate_accuracy": round(escalate_acc, 4),
        "escalate_precision": round(escalate_p, 4),
        "escalate_recall": round(escalate_r, 4),
        "escalate_f1": round(escalate_f1, 4),
        "n_pipeline_errors": int(n_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows (quick test)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    golden, has_baseline_cols = load_base_dataframe()
    if args.limit:
        golden = golden.head(args.limit).copy()
        print(f"--limit {args.limit}: only running the first {len(golden)} rows.")

    n_rows = len(golden)

    checkpoint = None if args.no_resume else load_checkpoint(n_rows)
    if checkpoint is not None:
        for col in ["pipeline_intent", "pipeline_all_intents", "pipeline_confidence",
                    "pipeline_reply", "pipeline_escalate", "pipeline_escalate_reason"]:
            golden[col] = checkpoint[col]
        golden["pipeline_escalate"] = golden["pipeline_escalate"].apply(_to_bool)
    else:
        for col in ["pipeline_intent", "pipeline_all_intents", "pipeline_confidence",
                    "pipeline_reply", "pipeline_escalate", "pipeline_escalate_reason"]:
            golden[col] = None

    print("\nSetting up retriever (builds ChromaDB index on first run, ~2 min)...")
    retriever = ReplyRetriever()
    retriever.build(AMAZON_PAIRS_CSV)

    todo_mask = golden["pipeline_intent"].isna()
    todo_idx = golden.index[todo_mask].tolist()
    print(f"\nRunning real pipeline over {len(todo_idx)}/{n_rows} remaining rows...")

    for i, idx in enumerate(tqdm(todo_idx, desc="  run_agent")):
        row = golden.loc[idx]
        pred = predict_one(row["customer_clean"], retriever)
        for k, v in pred.items():
            golden.at[idx, k] = v

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo_idx):
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            golden.to_csv(CHECKPOINT_PATH, index=False)

    golden["pipeline_escalate"] = golden["pipeline_escalate"].apply(_to_bool)

    metrics = evaluate(golden)

    golden.to_csv(BASELINE_PREDICTIONS, index=False)
    print(f"\nSaved predictions -> {BASELINE_PREDICTIONS}")
    if not has_baseline_cols:
        print("  (trivial_*/simple_* columns are absent -- run scripts/04_baselines.py "
              "to add them for the full 3-way comparison table.)")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "pipeline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"n_golden_rows": n_rows, "pipeline": metrics}, f, indent=2)
    print(f"Saved metrics     -> {metrics_path}")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"Removed checkpoint -> {CHECKPOINT_PATH}")

    print("\nNext: agent/metrics.py scores reply QUALITY (ROUGE-L, embedding "
          "similarity) on the pipeline_reply column this script just produced.")


if __name__ == "__main__":
    main()
