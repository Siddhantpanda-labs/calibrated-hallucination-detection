"""
Gray-box feature extraction:
Computes summary statistics from per-token log-probabilities and entropy
extracted from the greedy generation pass.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def compute_entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Computes Shannon entropy H = -sum(p * log(p)) from raw logits.
    logits: shape (seq_len, vocab_size) or (vocab_size,)
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Numerical stability via logsumexp
    max_logit = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp

    # H = log(sum_exp) - (sum(logits * exp_logits) / sum_exp) + max_logit
    # Standard: -sum(p * log(p)) with clipping
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    return entropy


def extract_graybox_features(
    token_logprobs: Sequence[float],
    token_entropies: Sequence[float] | None = None,
) -> dict[str, float]:
    """
    Extracts summary statistics from token logprobs and entropy for an answer.
    
    Features:
    - mean_logprob: average token log probability
    - min_logprob: minimum token log probability
    - first_token_logprob: log probability of the first generated token
    - mean_entropy: average token entropy across answer
    - max_entropy: maximum token entropy across answer
    - answer_length: number of generated tokens
    """
    lps = np.asarray(token_logprobs, dtype=float)
    T = len(lps)
    
    if T == 0:
        return {
            "mean_logprob": -10.0,
            "min_logprob": -10.0,
            "first_token_logprob": -10.0,
            "mean_entropy": 0.0,
            "max_entropy": 0.0,
            "answer_length": 0.0,
        }

    mean_lp = float(np.mean(lps))
    min_lp = float(np.min(lps))
    first_lp = float(lps[0])

    if token_entropies is not None and len(token_entropies) > 0:
        ents = np.asarray(token_entropies, dtype=float)
        mean_ent = float(np.mean(ents))
        max_ent = float(np.max(ents))
    else:
        # Approximate entropy from logprobs if token distributions were not retained
        # Note: -log(p) provides a local surprisal estimate
        mean_ent = float(-np.mean(lps))
        max_ent = float(-np.min(lps))

    return {
        "mean_logprob": mean_lp,
        "min_logprob": min_lp,
        "first_token_logprob": first_lp,
        "mean_entropy": mean_ent,
        "max_entropy": max_ent,
        "answer_length": float(T),
    }

