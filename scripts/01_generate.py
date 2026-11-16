"""
Step 1: Generation Script
Generates greedy answers, token logprobs, entropy, hidden states, and stochastic samples.
Saves to data/processed/generations_{dataset}.parquet.
"""

from __future__ import annotations

import os
import sys
import argparse
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from halludetect.config import load_config, safe_tqdm
from halludetect.data import load_triviaqa, load_truthfulqa, generate_mock_dataset, save_dataframe
from halludetect.generate import get_generator


def main():
    parser = argparse.ArgumentParser(description="Generate LLM answers and uncertainty traces")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "truthfulqa"], help="Dataset to run")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    parser.add_argument("--mock", action="store_true", help="Run with mock generator for dry-runs / testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    gen_cfg = cfg.get("generation", {})
    
    n_samples = args.num_samples
    if n_samples is None:
        n_samples = data_cfg.get("triviaqa_samples", 2000) if args.dataset == "triviaqa" else data_cfg.get("truthfulqa_samples", 817)

    output_path = args.output or f"data/processed/generations_{args.dataset}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"=== [Step 1: Generation] dataset={args.dataset}, num_samples={n_samples}, mock={args.mock} ===")

    # 1. Load data
    if args.mock:
        df_raw = generate_mock_dataset(num_samples=n_samples, dataset_type=args.dataset, seed=data_cfg.get("seed", 42))
    elif args.dataset == "triviaqa":
        df_raw = load_triviaqa(num_samples=n_samples, seed=data_cfg.get("seed", 42))
    else:
        df_raw = load_truthfulqa(num_samples=n_samples, seed=data_cfg.get("seed", 42))

    print(f"Loaded {len(df_raw)} questions.")

    # 2. Initialize generator
    generator = get_generator(config=cfg, mock=args.mock, seed=data_cfg.get("seed", 42))

    # 3. Generate answers and internal traces
    records = []
    num_samples_per_q = gen_cfg.get("num_samples", 5)

    for idx, row in safe_tqdm(df_raw.iterrows(), total=len(df_raw), desc="Generating"):
        q_id = row["question_id"]
        q_text = row["question"]
        gold_answers = row.get("gold_answers", [])

        if hasattr(generator, "generate"):
            if args.mock:
                res = generator.generate(q_id, q_text, gold_answers=gold_answers)
            else:
                res = generator.generate(
                    question_id=q_id,
                    question=q_text,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 32),
                    num_samples=num_samples_per_q,
                    temperature=gen_cfg.get("temperature", 0.7),
                    top_p=gen_cfg.get("top_p", 0.9),
                )

        rec = {
            "question_id": res.question_id,
            "question": res.question,
            "greedy_text": res.greedy_text,
            "token_logprobs": res.token_logprobs,
            "token_entropies": res.token_entropies,
            "hidden_states": res.hidden_states,
            "sampled_answers": res.sampled_answers,
            "gold_answers": gold_answers,
            "dataset": args.dataset,
        }
        if "correct_answers" in row:
            rec["correct_answers"] = row["correct_answers"]
        if "incorrect_answers" in row:
            rec["incorrect_answers"] = row["incorrect_answers"]

        records.append(rec)

    df_out = pd.DataFrame(records)
    save_dataframe(df_out, output_path)
    print(f"Successfully saved {len(df_out)} generations to {output_path}")


if __name__ == "__main__":
    main()
