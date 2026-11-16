"""
Evaluation metrics for hallucination detection and calibration.
Includes AUROC, AUPRC, ECE, Brier score, Risk-Coverage curves, and Bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Area Under the ROC Curve."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (Average Precision)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier score (mean squared error of probabilistic predictions)."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return float(brier_score_loss(y_true, y_prob))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[float, float, dict[str, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        y_true: Binary ground-truth labels (1 = hallucinated, 0 = correct)
        y_prob: Predicted probability of hallucination in [0, 1]
        n_bins: Number of bins
        strategy: 'uniform' (equal width) or 'quantile' (equal count)

    Returns:
        (ece, mce, bin_details)
        bin_details contains bin_edges, bin_confs, bin_accs, bin_counts
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0, 0.0, {}

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        
        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            diff = abs(acc - conf)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
        else:
            conf = 0.5 * (low + high)
            acc = 0.0
        
        bin_confs.append(conf)
        bin_accs.append(acc)
        bin_counts.append(count)

    details = {
        "bin_edges": bin_edges,
        "bin_confs": np.array(bin_confs),
        "bin_accs": np.array(bin_accs),
        "bin_counts": np.array(bin_counts),
    }

    return float(ece), float(mce), details


def compute_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes selective prediction risk-coverage curve.
    Higher y_prob implies higher risk of hallucination.
    We rank instances by confidence of being correct (1 - y_prob).
    For coverage fraction c in (0, 1], we answer the top c fraction
    and compute the risk (hallucination error rate) among answered queries.

    Returns:
        coverages: array of coverage points in (0, 1]
        risks: error rate at each coverage point
        aurc: Area Under the Risk-Coverage Curve
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return np.array([]), np.array([]), 0.0

    # Sort instances in ascending order of hallucination probability (most confident first)
    sorted_indices = np.argsort(y_prob)
    sorted_labels = y_true[sorted_indices]

    # Cumulative errors
    cum_errors = np.cumsum(sorted_labels)
    ks = np.arange(1, n + 1)
    empirical_risks = cum_errors / ks
    empirical_coverages = ks / n

    # Subsample or interpolate to n_points for consistent curves
    coverages = np.linspace(1.0 / n, 1.0, min(n_points, n))
    risks = np.interp(coverages, empirical_coverages, empirical_risks)

    # Calculate AURC via trapezoidal integration
    aurc = float(np.trapezoid(risks, coverages)) if hasattr(np, "trapezoid") else float(np.trapz(risks, coverages))

    return coverages, risks, aurc


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "auroc",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Computes bootstrap confidence interval for a given metric.
    
    Supported metric_name: 'auroc', 'auprc', 'ece', 'brier', 'aurc'.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    def calc(y_t, y_p):
        if metric_name == "auroc":
            return compute_auroc(y_t, y_p)
        elif metric_name == "auprc":
            return compute_auprc(y_t, y_p)
        elif metric_name == "ece":
            return compute_ece(y_t, y_p)[0]
        elif metric_name == "brier":
            return compute_brier_score(y_t, y_p)
        elif metric_name == "aurc":
            return compute_risk_coverage(y_t, y_p)[2]
        else:
            raise ValueError(f"Unknown metric {metric_name}")

    point_estimate = calc(y_true, y_prob)
    boot_values = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        val = calc(y_true[idx], y_prob[idx])
        if not np.isnan(val):
            boot_values.append(val)

    if not boot_values:
        return {
            "point": point_estimate,
            "std": 0.0,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
        }

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_values, alpha * 100))
    upper = float(np.percentile(boot_values, (1.0 - alpha) * 100))
    std = float(np.std(boot_values))

    return {
        "point": float(point_estimate),
        "std": std,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluates all metrics with bootstrap confidence intervals."""
    results = {}
    for metric in ["auroc", "auprc", "ece", "brier", "aurc"]:
        results[metric] = bootstrap_ci(
            y_true, y_prob, metric_name=metric, n_bootstraps=n_bootstraps, seed=seed
        )
    return results

