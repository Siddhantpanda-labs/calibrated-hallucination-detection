"""
Step 3: Feature Extraction Script
Extracts gray-box, black-box, and white-box internal features from generation traces.
Saves to data/processed/features_{dataset}.parquet.
"""

from __future__ import annotations

import os
import sys
import argparse
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from halludetect.config import load_config, safe_tqdm
from halludetect.data import load_dataframe, save_dataframe
from halludetect.features.grey import extract_graybox_features
from halludetect.features.consistency import extract_blackbox_features


def main():
    parser = argparse.ArgumentParser(description="Extract uncertainty features from generation outputs")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "truthfulqa"], help="Dataset name")
    parser.add_argument("--input", type=str, default=None, help="Input parquet path")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    input_path = args.input or f"data/processed/labeled_{args.dataset}.parquet"
    output_path = args.output or f"data/processed/features_{args.dataset}.parquet"

    print(f"=== [Step 3: Feature Extraction] dataset={args.dataset} ===")
    try:
        df = load_dataframe(input_path)
    except FileNotFoundError:
        print(f"Error: input file {input_path} does not exist. Please run scripts/02_label.py first.")
        sys.exit(1)
    print(f"Loaded {len(df)} records from {input_path}")

    records = []
    for _, row in safe_tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        greedy_text = row["greedy_text"]
        token_lps = row["token_logprobs"]
        token_ents = row.get("token_entropies", None)
        sampled_answers = row.get("sampled_answers", [])

        # 1. Gray-box features
        gray_feats = extract_graybox_features(token_lps, token_ents)

        # 2. Black-box features
        black_feats = extract_blackbox_features(greedy_text, sampled_answers)

        combined_rec = {
            "question_id": row["question_id"],
            "question": row["question"],
            "greedy_text": greedy_text,
            "label": row["label"],
            "dataset": row.get("dataset", args.dataset),
            **gray_feats,
            **black_feats,
            "hidden_states": {str(k): v for k, v in row.get("hidden_states", {}).items()} if isinstance(row.get("hidden_states"), dict) else {},
        }
        records.append(combined_rec)

    df_feats = pd.DataFrame(records)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except (OSError, PermissionError):
        pass
    save_dataframe(df_feats, output_path)
    print(f"Successfully extracted features for {len(df_feats)} records to {output_path}")


if __name__ == "__main__":
    main()
