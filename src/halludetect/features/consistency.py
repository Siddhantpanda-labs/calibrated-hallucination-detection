"""
Black-box feature extraction:
Self-consistency, agreement rate, semantic entropy, and clustering.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Callable
from collections import Counter
from halludetect.labeling import normalize_text, is_token_containment_match


def compute_agreement_rate(
    greedy_answer: str,
    sampled_answers: Sequence[str],
    matcher: Callable[[str, Sequence[str]], bool] | None = None,
) -> float:
    """
    Fraction of sampled answers that agree with the greedy answer.
    """
    if sampled_answers is None or len(sampled_answers) == 0:
        return 0.0

    if matcher is None:
        matcher = is_token_containment_match

    matches = sum(1 for sample in sampled_answers if matcher(sample, [greedy_answer]))
    return float(matches / len(sampled_answers))


def cluster_by_string(answers: Sequence[str]) -> list[list[int]]:
    """
    Clusters answers into equivalence groups based on normalized string / token overlap.
    Returns list of clusters, each cluster containing indices of answers.
    """
    if answers is None or len(answers) == 0:
        return []

    n = len(answers)
    clusters: list[list[int]] = []
    normalized = [normalize_text(a) for a in answers]

    for i in range(n):
        norm_i = normalized[i]
        assigned = False
        for cluster in clusters:
            rep_norm = normalized[cluster[0]]
            # If matches cluster representative
            if norm_i == rep_norm or (norm_i and rep_norm and is_token_containment_match(answers[i], [answers[cluster[0]]])):
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    return clusters


def cluster_by_nli(
    answers: Sequence[str],
    nli_predict_fn: Callable[[str, str], float],
    threshold: float = 0.5,
) -> list[list[int]]:
    """
    Clusters answers into semantic equivalence classes using bidirectional NLI.
    Two answers u and v are equivalent if:
    P(u entails v) > threshold AND P(v entails u) > threshold.
    """
    if answers is None or len(answers) == 0:
        return []

    n = len(answers)
    clusters: list[list[int]] = []

    for i in range(n):
        assigned = False
        for cluster in clusters:
            rep = answers[cluster[0]]
            cand = answers[i]
            # Fast check if strings are identical
            if normalize_text(rep) == normalize_text(cand):
                cluster.append(i)
                assigned = True
                break
            # Bidirectional entailment
            p_forward = nli_predict_fn(cand, rep)
            p_backward = nli_predict_fn(rep, cand)
            if p_forward >= threshold and p_backward >= threshold:
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    return clusters


def compute_semantic_entropy(
    sampled_answers: Sequence[str],
    clusters: list[list[int]] | None = None,
    nli_predict_fn: Callable[[str, str], float] | None = None,
) -> tuple[float, int]:
    """
    Computes semantic entropy across clusters:
    H = - sum_k (p_k * log(p_k))
    where p_k = |Cluster_k| / N.

    Returns:
        (semantic_entropy, num_clusters)
    """
    if sampled_answers is None or len(sampled_answers) == 0:
        return 0.0, 0

    N = len(sampled_answers)

    if clusters is None:
        if nli_predict_fn is not None:
            clusters = cluster_by_nli(sampled_answers, nli_predict_fn)
        else:
            clusters = cluster_by_string(sampled_answers)

    K = len(clusters)
    if K <= 1:
        return 0.0, K

    probs = np.array([len(c) / N for c in clusters], dtype=float)
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy, K


def extract_blackbox_features(
    greedy_answer: str,
    sampled_answers: Sequence[str],
    nli_predict_fn: Callable[[str, str], float] | None = None,
) -> dict[str, float]:
    """
    Extracts black-box uncertainty features:
    - agreement_rate: fraction of samples matching greedy answer
    - semantic_entropy: entropy over semantic clusters
    - num_semantic_clusters: number of distinct semantic clusters
    """
    if sampled_answers is None or len(sampled_answers) == 0:
        return {
            "agreement_rate": 0.0,
            "semantic_entropy": 0.0,
            "num_semantic_clusters": 1.0,
        }

    agr = compute_agreement_rate(greedy_answer, sampled_answers)
    sem_ent, num_clusters = compute_semantic_entropy(
        sampled_answers, nli_predict_fn=nli_predict_fn
    )

    return {
        "agreement_rate": agr,
        "semantic_entropy": sem_ent,
        "num_semantic_clusters": float(num_clusters),
    }

