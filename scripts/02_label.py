"""
Step 2: Labeling Script
Labels greedy answers against ground-truth targets (y = 1 for hallucinated, 0 for correct).
Audits labels on 100 questions to quantify label error rate and prevent silent errors.
Saves to data/processed/labeled_{dataset}.parquet.
"""

from __future__ import annotations

import os
import sys
import argparse
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from halludetect.data import load_dataframe, save_dataframe
from halludetect.labeling import (
    label_prediction,
    label_truthfulqa_prediction,
    audit_labeling,
)


def main():
    parser = argparse.ArgumentParser(description="Label model generations against gold answers")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "truthfulqa"], help="Dataset to label")
    parser.add_argument("--input", type=str, default=None, help="Input parquet path")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    parser.add_argument("--audit-size", type=int, default=100, help="Number of samples to audit")
    args = parser.parse_args()

    input_path = args.input or f"data/processed/generations_{args.dataset}.parquet"
    output_path = args.output or f"data/processed/labeled_{args.dataset}.parquet"

    print(f"=== [Step 2: Labeling] dataset={args.dataset} ===")
    try:
        df = load_dataframe(input_path)
    except FileNotFoundError:
        print(f"Error: input file {input_path} does not exist. Please run scripts/01_generate.py first.")
        sys.exit(1)
    print(f"Loaded {len(df)} generations from {input_path}")

    # Assign labels
    labels = []
    for _, row in df.iterrows():
        greedy_text = row["greedy_text"]
        
        if args.dataset == "truthfulqa" and "correct_answers" in row:
            correct = row["correct_answers"]
            incorrect = row.get("incorrect_answers", [])
            label = label_truthfulqa_prediction(greedy_text, correct, incorrect)
        else:
            golds = row.get("gold_answers", [])
            label = label_prediction(greedy_text, golds, mode="token_containment")

        labels.append(label)

    df["label"] = labels
    hallu_count = sum(labels)
    total_count = len(labels)
    hallu_rate = hallu_count / total_count if total_count > 0 else 0.0

    print(f"Labeled {total_count} examples: {hallu_count} hallucinated ({hallu_rate * 100:.1f}%), {total_count - hallu_count} correct.")

    # Run audit
    os.makedirs("results/tables", exist_ok=True)
    audit_records = audit_labeling(
        predictions=df["greedy_text"].tolist(),
        gold_answers_list=df["gold_answers"].tolist(),
        labels=df["label"].tolist(),
        sample_size=args.audit_size,
    )
    df_audit = pd.DataFrame(audit_records)
    audit_path = f"results/tables/label_audit_{args.dataset}.csv"
    df_audit.to_csv(audit_path, index=False)
    
    suspicious_count = df_audit["suspicious"].sum()
    print(f"Audit completed ({len(df_audit)} samples): {suspicious_count} potentially noisy labels flagged ({suspicious_count / len(df_audit) * 100:.1f}%).")
    print(f"Audit report written to {audit_path}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except (OSError, PermissionError):
        pass
    save_dataframe(df, output_path)
    print(f"Successfully saved labeled data to {output_path}")


if __name__ == "__main__":
    main()
