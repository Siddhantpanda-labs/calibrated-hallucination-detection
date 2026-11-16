"""
Visualization and plotting module for hallucination detection.
Produces publication-ready figures:
1. Reliability diagrams (calibrated vs uncalibrated)
2. Layer sweep AUROC curve for linear probes
3. Risk-coverage curves for selective prediction
4. ROC and Precision-Recall curves
5. Cost-performance trade-off comparison
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Any
from sklearn.metrics import roc_curve, precision_recall_curve
from halludetect.metrics import compute_ece, compute_risk_coverage, compute_auroc, compute_auprc


# Style configuration for clean, modern academic plots
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_reliability_diagram(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    n_bins: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plots side-by-side reliability diagrams with accuracy vs confidence
    and sample frequency histograms.
    """
    n_models = len(probs_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (name, probs) in zip(axes, probs_dict.items()):
        ece, mce, details = compute_ece(y_true, probs, n_bins=n_bins)
        confs = details["bin_confs"]
        accs = details["bin_accs"]
        edges = details["bin_edges"]
        widths = np.diff(edges)

        # Plot perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")
        
        # Plot empirical bar plot
        ax.bar(
            edges[:-1],
            accs,
            width=widths,
            align="edge",
            alpha=0.6,
            edgecolor="black",
            color="#2b5c8f",
            label="Outputs",
        )
        
        # Plot gap
        gap = np.abs(accs - confs)
        ax.bar(
            edges[:-1],
            gap,
            bottom=np.minimum(accs, confs),
            width=widths,
            align="edge",
            alpha=0.3,
            color="#d9534f",
            hatch="//",
            label="Calibration gap",
        )

        ax.set_title(f"{name}\nECE = {ece:.3f} | MCE = {mce:.3f}")
        ax.set_xlabel("Confidence P(hallucinated)")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper left")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig


def plot_layer_sweep(
    sweep_results: Sequence[dict[str, Any]],
    best_layer: int | str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plots probe AUROC across transformer layers.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    layers = [r["layer"] for r in sweep_results]
    aurocs = [r["auroc"] for r in sweep_results]

    x_indices = np.arange(len(layers))
    ax.plot(x_indices, aurocs, marker="o", linewidth=2.2, color="#1f77b4", label="Probe AUROC")

    if best_layer is not None:
        best_idx = layers.index(best_layer)
        best_auroc = aurocs[best_idx]
        ax.scatter(
            [best_idx],
            [best_auroc],
            color="#d62728",
            s=120,
            zorder=5,
            label=f"Best Layer ({best_layer}): {best_auroc:.3f}",
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Linear Probe AUROC")
    ax.set_title("Layer-Sweep Probe Discrimination")
    ax.set_ylim([0.45, max(1.0, max(aurocs) + 0.05)])
    ax.legend(loc="lower right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig


def plot_risk_coverage_curve(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plots Selective Prediction Risk-Coverage curves.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, probs) in enumerate(probs_dict.items()):
        coverages, risks, aurc = compute_risk_coverage(y_true, probs)
        color = colors[idx % len(colors)]
        ax.plot(
            coverages,
            risks,
            label=f"{name} (AURC = {aurc:.3f})",
            linewidth=2.0,
            color=color,
        )

    ax.set_xlabel("Coverage (Fraction of questions answered)")
    ax.set_ylabel("Risk (Error rate on answered set)")
    ax.set_title("Selective Answering: Risk vs. Coverage")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, max(0.5, float(np.mean(y_true)) + 0.1)])
    ax.legend(loc="upper left")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plots ROC and PR curves side-by-side.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, probs) in enumerate(probs_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec, rec, _ = precision_recall_curve(y_true, probs)
        auroc = compute_auroc(y_true, probs)
        auprc = compute_auprc(y_true, probs)

        color = colors[idx % len(colors)]
        ax_roc.plot(fpr, tpr, label=f"{name} ({auroc:.3f})", linewidth=1.8, color=color)
        ax_pr.plot(rec, prec, label=f"{name} ({auprc:.3f})", linewidth=1.8, color=color)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_title("ROC Curves (Hallucination Detection)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    base_rate = float(np.mean(y_true))
    ax_pr.axhline(base_rate, color="k", linestyle="--", alpha=0.5, label=f"Random ({base_rate:.2f})")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig


def plot_cost_performance_frontier(
    methods: Sequence[dict[str, Any]],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plots inference cost (e.g. additional forward passes) vs AUROC.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for m in methods:
        name = m["name"]
        cost = m["extra_forward_passes"]
        auroc = m["auroc"]
        family = m.get("family", "gray")

        color = "#1f77b4" if family == "gray" else ("#2ca02c" if family == "black" else "#d62728")
        ax.scatter([cost], [auroc], s=120, color=color, zorder=5)
        ax.annotate(
            name,
            (cost, auroc),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=10,
        )

    ax.set_xlabel("Extra Forward Passes per Query (Inference Cost)")
    ax.set_ylabel("Discrimination AUROC")
    ax.set_title("Cost vs. Performance Frontier")
    ax.set_xlim([-0.5, max(m["extra_forward_passes"] for m in methods) + 2])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return fig

