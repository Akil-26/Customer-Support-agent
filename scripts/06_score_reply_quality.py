"""
Step 6 (part 2) -- Score reply QUALITY (ROUGE-L + embedding similarity)
for every system present in data/eval/baseline_predictions.csv, using
agent/metrics.py. Credential-free (no LLM call) -- only needs the *_reply
columns that 04_baselines.py and/or 05_run_pipeline_on_golden.py already
produced.

Reference for every comparison = golden_set.csv's own brand_reply_clean:
the real historical Amazon reply to that specific customer message (see
agent/metrics.py's docstring for the caveat this implies).

Also produces the mandatory-for-the-report worked example: a real row
where a good reply scores low on ROUGE-L because it paraphrases instead
of copying wording, found via agent.metrics.find_rouge_failure_examples()
run on the REAL pipeline's own replies (not a hypothetical).

Usage:
    python scripts/06_score_reply_quality.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from agent.metrics import batch_reply_quality, find_rouge_failure_examples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "eval", "baseline_predictions.csv")
REPORTS_DIR = os.path.join(ROOT, "reports")

# (column_name, display_label) -- only scored if the column is present,
# so this works whether you've run 04 only, 05 only, or both.
SYSTEMS = [
    ("trivial_reply", "Trivial baseline"),
    ("simple_reply", "Simple baseline"),
    ("pipeline_reply", "Real pipeline"),
]


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run scripts/04_baselines.py and/or scripts/05_run_pipeline_on_golden.py first.")
        sys.exit(1)

    df = pd.read_csv(PREDICTIONS_PATH)
    reference = df["brand_reply_clean"]

    present_systems = [(col, label) for col, label in SYSTEMS if col in df.columns]
    if not present_systems:
        print("ERROR: none of trivial_reply / simple_reply / pipeline_reply columns found "
              f"in {PREDICTIONS_PATH}. Run 04_baselines.py and/or 05_run_pipeline_on_golden.py first.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows. Scoring: {', '.join(label for _, label in present_systems)}\n")

    results = {}
    print(f"{'System':<20}{'Avg ROUGE-L':>14}{'Avg Embed Sim':>16}")
    for col, label in present_systems:
        per_row = batch_reply_quality(reference.tolist(), df[col].tolist())
        avg_rouge = sum(r["rouge_l"] for r in per_row) / len(per_row)
        avg_embed = sum(r["embedding_similarity"] for r in per_row) / len(per_row)
        results[col] = {
            "label": label,
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_embedding_similarity": round(avg_embed, 4),
            "n_scored": sum(1 for r, h in zip(reference, df[col]) if str(r).strip() and str(h).strip()),
            "n_total": len(df),
        }
        print(f"{label:<20}{avg_rouge:>14.3f}{avg_embed:>16.3f}")

    worked_example = None
    if "pipeline_reply" in df.columns:
        examples = find_rouge_failure_examples(reference.tolist(), df["pipeline_reply"].tolist(), top_n=3)
        if examples:
            worked_example = examples[0]
            print("\n--- Worked example: why ROUGE-L alone is misleading (real data) ---")
            print(f"  Reference (actual historical reply): {worked_example['reference'][:150]!r}")
            print(f"  Pipeline reply (paraphrase):          {worked_example['hypothesis'][:150]!r}")
            print(f"  ROUGE-L: {worked_example['rouge_l']:.3f}   |   Embedding similarity: {worked_example['embedding_similarity']:.3f}")
        else:
            print("\n(No non-empty pipeline_reply rows to build a worked example from yet -- "
                  "most rows may have escalated. Try scripts/05 with --limit on a larger sample, "
                  "or check the escalate column.)")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "reply_quality_metrics.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_rows": len(df),
            "systems": results,
            "rouge_failure_worked_example": worked_example,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
