"""
Dataset loading, parsing, splitting, and mock generation.
Supports TriviaQA (in-domain) and TruthfulQA (out-of-distribution).
Strictly enforces question-level train/cal/test splitting without data leakage.
"""

from __future__ import annotations

import os
import hashlib
import numpy as np
import pandas as pd
from typing import Sequence, Any


def save_dataframe(df: pd.DataFrame, path: str):
    """Saves dataframe as parquet if engine available, or falls back to pickle."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, PermissionError):
        pass

    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        alt_path = path if path.endswith(".pkl") else path + ".pkl"
        df.to_pickle(alt_path)


def load_dataframe(path: str) -> pd.DataFrame:
    """Loads dataframe from parquet or fallback pickle."""
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError):
            pass
    alt_path = path if path.endswith(".pkl") else path + ".pkl"
    if os.path.exists(alt_path):
        return pd.read_pickle(alt_path)
    raise FileNotFoundError(f"Neither {path} nor {alt_path} found.")


def split_by_question(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    cal_frac: float = 0.15,
    test_frac: float = 0.25,
    id_col: str = "question_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe by unique question ID into train, calibration, and test splits.
    Guarantees no question ID is shared across splits (zero leakage).
    """
    assert abs((train_frac + cal_frac + test_frac) - 1.0) < 1e-5, "Fractions must sum to 1.0"
    
    unique_ids = df[id_col].drop_duplicates().values
    n = len(unique_ids)
    
    rng = np.random.RandomState(seed)
    shuffled_ids = rng.permutation(unique_ids)
    
    n_train = int(np.round(n * train_frac))
    n_cal = int(np.round(n * cal_frac))
    
    train_ids = set(shuffled_ids[:n_train])
    cal_ids = set(shuffled_ids[n_train:n_train + n_cal])
    test_ids = set(shuffled_ids[n_train + n_cal:])
    
    # Assert zero leakage
    assert len(train_ids & cal_ids) == 0, "Train and Cal splits overlap!"
    assert len(train_ids & test_ids) == 0, "Train and Test splits overlap!"
    assert len(cal_ids & test_ids) == 0, "Cal and Test splits overlap!"
    
    df_train = df[df[id_col].isin(train_ids)].copy().reset_index(drop=True)
    df_cal = df[df[id_col].isin(cal_ids)].copy().reset_index(drop=True)
    df_test = df[df[id_col].isin(test_ids)].copy().reset_index(drop=True)
    
    return df_train, df_cal, df_test


def load_triviaqa(
    num_samples: int | None = 2000,
    split: str = "train",
    cache_dir: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Loads TriviaQA from HuggingFace datasets or local parquet cache.
    Extracts: question_id, question, gold_answers (list of string aliases).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.nocontext", split=split, cache_dir=cache_dir)
        if num_samples is not None and num_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(num_samples))
        
        records = []
        for item in ds:
            q_id = item.get("question_id", str(item.get("id", "")))
            q_text = item.get("question", "")
            ans_info = item.get("answer", {})
            value = ans_info.get("value", "")
            aliases = list(ans_info.get("aliases", []))
            normalized_aliases = list(ans_info.get("normalized_aliases", []))
            
            all_answers = list(dict.fromkeys([value] + aliases + normalized_aliases))
            all_answers = [a for a in all_answers if a]
            
            records.append({
                "question_id": q_id,
                "question": q_text,
                "gold_answers": all_answers,
                "dataset": "triviaqa",
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[load_triviaqa] Notice: Falling back to local mock generator ({e})")
        return generate_mock_dataset(num_samples or 100, dataset_type="triviaqa", seed=seed)


def load_truthfulqa(
    num_samples: int | None = 817,
    cache_dir: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Loads TruthfulQA from HuggingFace datasets.
    Extracts: question_id, question, correct_answers, incorrect_answers.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation", split="validation", cache_dir=cache_dir)
        if num_samples is not None and num_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(num_samples))

        records = []
        for idx, item in enumerate(ds):
            q_text = item.get("question", "")
            best_ans = item.get("best_answer", "")
            correct = list(item.get("correct_answers", []))
            incorrect = list(item.get("incorrect_answers", []))
            
            records.append({
                "question_id": f"tqa_{idx}",
                "question": q_text,
                "gold_answers": list(dict.fromkeys([best_ans] + correct)),
                "correct_answers": correct,
                "incorrect_answers": incorrect,
                "dataset": "truthfulqa",
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[load_truthfulqa] Notice: Falling back to local mock generator ({e})")
        return generate_mock_dataset(num_samples or 100, dataset_type="truthfulqa", seed=seed)


def generate_mock_dataset(
    num_samples: int = 100,
    dataset_type: str = "triviaqa",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates deterministic synthetic QA data for unit tests and offline pipeline verification.
    """
    rng = np.random.RandomState(seed)
    
    mock_pool = [
        ("What is the capital of France?", ["Paris", "City of Paris"]),
        ("Who wrote Romeo and Juliet?", ["William Shakespeare", "Shakespeare"]),
        ("What is the chemical symbol for gold?", ["Au", "gold"]),
        ("Which planet is known as the Red Planet?", ["Mars", "red planet"]),
        ("What is the largest ocean on Earth?", ["Pacific Ocean", "Pacific"]),
        ("Who developed the theory of general relativity?", ["Albert Einstein", "Einstein"]),
        ("What is the square root of 64?", ["8", "eight"]),
        ("What is the boiling point of water in Celsius?", ["100", "100 C", "100 degrees"]),
        ("In what year did Apollo 11 land on the moon?", ["1969"]),
        ("What is the tallest mountain in the world?", ["Mount Everest", "Everest"]),
    ]

    records = []
    for i in range(num_samples):
        base_q, base_answers = mock_pool[i % len(mock_pool)]
        q_id = f"{dataset_type}_{seed}_{i:05d}"
        q_text = f"{base_q} (Variant {i // len(mock_pool)})" if i >= len(mock_pool) else base_q
        
        rec = {
            "question_id": q_id,
            "question": q_text,
            "gold_answers": list(base_answers),
            "dataset": dataset_type,
        }
        if dataset_type == "truthfulqa":
            rec["correct_answers"] = list(base_answers)
            rec["incorrect_answers"] = ["Unknown", "False answer"]

        records.append(rec)

    return pd.DataFrame(records)
