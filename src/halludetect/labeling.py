"""
Labeling module for hallucination detection.
Normalizes LLM generations, checks against gold answers and aliases,
and assigns binary labels: y = 1 (hallucinated/wrong), y = 0 (correct).
"""

from __future__ import annotations

import re
import string
from typing import Any, Iterable, Sequence


def normalize_text(text: str | None) -> str:
    """
    Standard QA text normalization:
    - Lowercase
    - Remove punctuation
    - Remove English articles ('a', 'an', 'the')
    - Collapse extra whitespace
    """
    if text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove articles as standalone words
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Collapse multiple whitespaces and strip
    text = " ".join(text.split())
    return text.strip()


def get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    norm = normalize_text(text)
    return norm.split() if norm else []


def is_exact_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    """Checks if normalized prediction strictly matches any normalized gold answer."""
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return False
    
    for gold in gold_answers:
        if norm_pred == normalize_text(gold):
            return True
    return False


def is_token_containment_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    """
    Checks if prediction tokens are contained within gold answer or vice-versa
    using word-boundary matching (prevents 'car' matching 'carpet').
    Also handles common short answers (e.g. 'Shakespeare' vs 'William Shakespeare').
    """
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return False
    
    pred_tokens = set(get_tokens(norm_pred))
    if not pred_tokens:
        return False

    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if not norm_gold:
            continue
        gold_tokens = set(get_tokens(norm_gold))
        if not gold_tokens:
            continue
        
        # Exact string equality after normalization
        if norm_pred == norm_gold:
            return True
        
        # Word-boundary substring match
        pattern_pred = r"\b" + re.escape(norm_pred) + r"\b"
        if re.search(pattern_pred, norm_gold):
            return True
        
        pattern_gold = r"\b" + re.escape(norm_gold) + r"\b"
        if re.search(pattern_gold, norm_pred):
            return True

        # If prediction has multiple tokens and is a subset of gold tokens (e.g., 'United States' in 'United States of America')
        if len(pred_tokens) >= 2 and pred_tokens.issubset(gold_tokens):
            return True

    return False


def label_prediction(
    prediction: str,
    gold_answers: Sequence[str],
    mode: str = "token_containment",
) -> int:
    """
    Assigns binary label:
    - 0: Correct
    - 1: Hallucinated / Incorrect

    Args:
        prediction: Generated string from LLM
        gold_answers: List of ground-truth target answers/aliases
        mode: 'exact' or 'token_containment' (default)
    """
    if not prediction or gold_answers is None or len(gold_answers) == 0:
        return 1

    if mode == "exact":
        is_correct = is_exact_match(prediction, gold_answers)
    else:
        is_correct = is_token_containment_match(prediction, gold_answers)

    return 0 if is_correct else 1


def label_truthfulqa_prediction(
    prediction: str,
    correct_answers: Sequence[str],
    incorrect_answers: Sequence[str] | None = None,
) -> int:
    """
    Labels a TruthfulQA response.
    TruthfulQA provides sets of correct and incorrect statements/answers.
    Returns:
        0: Correct (matches correct answers, doesn't match incorrect)
        1: Hallucinated (matches incorrect or fails to match correct)
    """
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return 1
    
    # Check if prediction directly matches any incorrect answer
    if incorrect_answers is not None and len(incorrect_answers) > 0:
        for wrong in incorrect_answers:
            if is_token_containment_match(prediction, [wrong]):
                return 1

    # Check if it matches any correct answer
    if correct_answers is not None and len(correct_answers) > 0:
        if is_token_containment_match(prediction, correct_answers):
            return 0

    return 1


def audit_labeling(
    predictions: Sequence[str],
    gold_answers_list: Sequence[Sequence[str]],
    labels: Sequence[int],
    sample_size: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Extracts an audit sample to inspect false positives / false negatives.
    Flags potential label noise (e.g. labeled wrong but high Jaccard token overlap).
    """
    import random
    rng = random.Random(seed)
    
    n = len(predictions)
    indices = list(range(n))
    if n > sample_size:
        indices = rng.sample(indices, sample_size)
    
    audit_records = []
    for idx in indices:
        pred = predictions[idx]
        golds = gold_answers_list[idx]
        label = labels[idx]
        
        pred_set = set(get_tokens(pred))
        max_overlap = 0.0
        for gold in golds:
            gold_set = set(get_tokens(gold))
            if gold_set:
                overlap = len(pred_set & gold_set) / len(pred_set | gold_set) if (pred_set | gold_set) else 0.0
                if overlap > max_overlap:
                    max_overlap = overlap

        # Flag potential mismatch: high overlap but marked wrong, or zero overlap but marked correct
        suspicious = (label == 1 and max_overlap > 0.6) or (label == 0 and max_overlap == 0.0)
        
        audit_records.append({
            "index": idx,
            "prediction": pred,
            "gold_answers": list(golds),
            "assigned_label": int(label),
            "max_jaccard_overlap": round(max_overlap, 3),
            "suspicious": suspicious,
        })
        
    return audit_records

