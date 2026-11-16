"""
Step 4: Training, Calibration, and Evaluation Script
Executes train/cal/test splits by question (zero leakage).
Evaluates baselines, linear probes, supervised models, and calibration methods.
Computes AUROC, AUPRC, ECE, Brier score, risk-coverage with 95% bootstrap CIs.
Evaluates cross-dataset generalization on TruthfulQA.
Exports comprehensive tables to results/tables/.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from halludetect.config import load_config, df_to_markdown
from halludetect.data import split_by_question, load_dataframe
from halludetect.features.probe import HiddenStateProbe, evaluate_layer_sweep
from halludetect.models import (
    create_baseline_logprob,
    create_baseline_agreement,
    create_baseline_semantic_entropy,
    SupervisedDetector,
    GRAYBOX_FEATURES,
    BLACKBOX_FEATURES,
    ALL_FEATURES,
)
from halludetect.calibration import CalibratedDetector
from halludetect.metrics import compute_all_metrics, compute_risk_coverage


def format_ci_metric(res: dict[str, float]) -> str:
    """Formats point estimate with 95% CI: point (lower - upper)."""
    pt = res["point"]
    lo = res["ci_lower"]
    hi = res["ci_upper"]
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Train, calibrate, and evaluate hallucination detectors")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--input-in-domain", type=str, default="data/processed/features_triviaqa.parquet", help="In-domain features")
    parser.add_argument("--input-ood", type=str, default="data/processed/features_truthfulqa.parquet", help="OOD features")
    parser.add_argument("--bootstraps", type=int, default=500, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"=== [Step 4: Train & Evaluate] seed={args.seed}, bootstraps={args.bootstraps} ===")
    try:
        df_id = load_dataframe(args.input_in_domain)
    except FileNotFoundError:
        print(f"Error: {args.input_in_domain} not found. Run scripts 01-03 first.")
        sys.exit(1)
    print(f"Loaded {len(df_id)} in-domain examples from {args.input_in_domain}")

    # 1. Question-level Split
    df_train, df_cal, df_test = split_by_question(
        df_id, train_frac=0.60, cal_frac=0.15, test_frac=0.25, seed=args.seed
    )
    print(f"Split sizes: Train={len(df_train)}, Cal={len(df_cal)}, Test={len(df_test)}")

    y_train = df_train["label"].values
    y_cal = df_cal["label"].values
    y_test = df_test["label"].values

    # Check for OOD data
    try:
        df_ood = load_dataframe(args.input_ood)
        has_ood = True
        y_ood = df_ood["label"].values
        print(f"Loaded {len(df_ood)} OOD (TruthfulQA) examples.")
    except FileNotFoundError:
        has_ood = False
        df_ood = None
        y_ood = None
        print("Note: OOD features not found. Skipping TruthfulQA cross-dataset test.")

    # 2. Layer-Sweep on White-Box Hidden States
    print("\n--- Sweeping Hidden-State Layers for Linear Probe ---")
    layer_states_train = {}
    layer_states_cal = {}
    layer_states_test = {}
    layer_states_ood = {}

    sample_hs = df_train["hidden_states"].iloc[0]
    available_layers = list(sample_hs.keys()) if isinstance(sample_hs, dict) else []

    for l_key in available_layers:
        layer_states_train[l_key] = np.array([row[l_key] for row in df_train["hidden_states"]])
        layer_states_cal[l_key] = np.array([row[l_key] for row in df_cal["hidden_states"]])
        layer_states_test[l_key] = np.array([row[l_key] for row in df_test["hidden_states"]])
        if has_ood and df_ood is not None:
            def _get_hs(row, key):
                if key in row:
                    return row[key]
                try:
                    int_k = int(key)
                    if int_k in row:
                        return row[int_k]
                except ValueError:
                    pass
                return row.get(str(key), None)
            layer_states_ood[l_key] = np.array([_get_hs(row, l_key) for row in df_ood["hidden_states"]])

    sweep_res = evaluate_layer_sweep(
        layer_states_train, y_train, layer_states_cal, y_cal, random_state=args.seed
    )
    best_layer = sweep_res["best_layer"]
    print(f"Layer sweep complete. Best layer: {best_layer} (Cal AUROC: {sweep_res['best_auroc']:.3f})")

    os.makedirs("results/tables", exist_ok=True)
    df_sweep = pd.DataFrame(sweep_res["sweep"])
    df_sweep.to_csv("results/tables/layer_sweep.csv", index=False)
    with open("results/tables/best_layer.json", "w") as f:
        json.dump({"best_layer": str(best_layer), "best_auroc": sweep_res["best_auroc"]}, f, indent=2)

    # Train probe on best layer
    best_probe = sweep_res["probes"][best_layer]
    probe_probs_test = best_probe.predict_proba(layer_states_test[best_layer])
    probe_probs_cal = best_probe.predict_proba(layer_states_cal[best_layer])
    probe_probs_train = best_probe.predict_proba(layer_states_train[best_layer])
    if has_ood:
        probe_probs_ood = best_probe.predict_proba(layer_states_ood[best_layer])

    # Augment dataframes with probe score feature
    df_train = df_train.copy()
    df_cal = df_cal.copy()
    df_test = df_test.copy()
    df_train["probe_score"] = probe_probs_train
    df_cal["probe_score"] = probe_probs_cal
    df_test["probe_score"] = probe_probs_test
    if has_ood and df_ood is not None:
        df_ood = df_ood.copy()
        df_ood["probe_score"] = probe_probs_ood

    # 3. Model Dictionary Setup
    models_to_evaluate = {
        # Unsupervised Baselines
        "Baseline: Mean LogProb": create_baseline_logprob(),
        "Baseline: Agreement Rate": create_baseline_agreement(),
        "Baseline: Semantic Entropy": create_baseline_semantic_entropy(),
        # Single Modality Supervised
        "Gray-box (LogReg)": SupervisedDetector("logistic", feature_names=GRAYBOX_FEATURES, random_state=args.seed),
        "Black-box (LogReg)": SupervisedDetector("logistic", feature_names=BLACKBOX_FEATURES, random_state=args.seed),
        "White-box Probe": best_probe,
        # Multi-Signal Combinations
        "Gray + Black (LogReg)": SupervisedDetector("logistic", feature_names=ALL_FEATURES, random_state=args.seed),
        "Gray + Black (GBM)": SupervisedDetector("gbm", feature_names=ALL_FEATURES, random_state=args.seed),
        "Combined: Gray+Black+Probe (LogReg)": SupervisedDetector(
            "logistic", feature_names=ALL_FEATURES + ["probe_score"], random_state=args.seed
        ),
        "Combined: Gray+Black+Probe (GBM)": SupervisedDetector(
            "gbm", feature_names=ALL_FEATURES + ["probe_score"], random_state=args.seed
        ),
    }

    # 4. Evaluation Loop
    print("\n--- Evaluating Models on In-Domain Test Set ---")
    results_rows = []
    test_predictions = {}
    calibrated_predictions = {}

    for name, model in models_to_evaluate.items():
        # Fit base model
        if name == "White-box Probe":
            raw_p_test = probe_probs_test
            raw_p_cal = probe_probs_cal
            raw_p_ood = probe_probs_ood if has_ood else None
        elif isinstance(model, SupervisedDetector):
            model.fit(df_train, y_train)
            raw_p_test = model.predict_proba(df_test)[:, 1]
            raw_p_cal = model.predict_proba(df_cal)[:, 1]
            raw_p_ood = model.predict_proba(df_ood)[:, 1] if has_ood else None
        else: # BaselineDetector
            model.fit(df_train)
            raw_p_test = model.predict_proba(df_test)[:, 1]
            raw_p_cal = model.predict_proba(df_cal)[:, 1]
            raw_p_ood = model.predict_proba(df_ood)[:, 1] if has_ood else None

        test_predictions[name] = raw_p_test

        # Compute In-Domain Raw Metrics with Bootstrap CIs
        metrics = compute_all_metrics(y_test, raw_p_test, n_bootstraps=args.bootstraps, seed=args.seed)

        # Fit Calibrators on Calibration Split
        calibrator_platt = CalibratedDetector(model, method="platt")
        calibrator_platt.calibrator.fit(raw_p_cal, y_cal)
        p_test_platt = calibrator_platt.calibrator.predict_proba(raw_p_test)
        metrics_platt = compute_all_metrics(y_test, p_test_platt, n_bootstraps=args.bootstraps, seed=args.seed)

        calibrator_iso = CalibratedDetector(model, method="isotonic")
        calibrator_iso.calibrator.fit(raw_p_cal, y_cal)
        p_test_iso = calibrator_iso.calibrator.predict_proba(raw_p_test)
        metrics_iso = compute_all_metrics(y_test, p_test_iso, n_bootstraps=args.bootstraps, seed=args.seed)

        calibrated_predictions[f"{name} (Platt)"] = p_test_platt

        # Evaluate OOD if available
        ood_auroc_str = "N/A"
        ood_ece_str = "N/A"
        if has_ood and raw_p_ood is not None:
            ood_metrics = compute_all_metrics(y_ood, raw_p_ood, n_bootstraps=args.bootstraps, seed=args.seed)
            ood_auroc_str = format_ci_metric(ood_metrics["auroc"])
            ood_ece_str = format_ci_metric(ood_metrics["ece"])

        results_rows.append({
            "Method": name,
            "AUROC (Test)": format_ci_metric(metrics["auroc"]),
            "AUPRC (Test)": format_ci_metric(metrics["auprc"]),
            "Raw ECE": format_ci_metric(metrics["ece"]),
            "Platt ECE": format_ci_metric(metrics_platt["ece"]),
            "Isotonic ECE": format_ci_metric(metrics_iso["ece"]),
            "Brier Score": format_ci_metric(metrics_platt["brier"]),
            "AURC (Risk-Coverage)": format_ci_metric(metrics["aurc"]),
            "OOD AUROC (TruthfulQA)": ood_auroc_str,
            "OOD ECE": ood_ece_str,
            # Point estimates for easy sorting/plotting
            "_auroc": metrics["auroc"]["point"],
            "_ece_raw": metrics["ece"]["point"],
            "_ece_platt": metrics_platt["ece"]["point"],
        })

    # Save summary tables
    df_results = pd.DataFrame(results_rows)
    df_results.sort_values(by="_auroc", ascending=False, inplace=True)
    
    clean_cols = [c for c in df_results.columns if not c.startswith("_")]
    df_results[clean_cols].to_csv("results/tables/evaluation_metrics.csv", index=False)
    
    # Save Markdown table
    md_table = df_to_markdown(df_results[clean_cols])
    with open("results/tables/evaluation_metrics.md", "w") as f:
        f.write("# Calibrated Hallucination Detection - Evaluation Results\n\n")
        f.write(f"In-domain: TriviaQA (Test N={len(y_test)}) | OOD: TruthfulQA\n\n")
        f.write(md_table + "\n")

    print("\n=== Evaluation Results Summary ===")
    print(md_table)

    # Save predictions for figure generation
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        "data/processed/predictions_cache.npz",
        y_test=y_test,
        **{f"pred__{k}": v for k, v in test_predictions.items()},
        **{f"cal__{k}": v for k, v in calibrated_predictions.items()},
    )
    print("\nSaved predictions cache to data/processed/predictions_cache.npz")


if __name__ == "__main__":
    main()
