"""
White-box probe module:
Extracts hidden states from transformer layers, trains linear probes,
and performs layer-sweep evaluation to identify optimal representation layers.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from halludetect.metrics import compute_auroc


class HiddenStateProbe:
    """
    Linear Probe trained on hidden states extracted from LLM internal layers.
    Standardizes representations and fits an L2-regularized logistic regression.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> HiddenStateProbe:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) > 1:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Fallback to neutral 0.5
            return np.full(len(X), 0.5, dtype=float)
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def score_auroc(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        return compute_auroc(y, probs)


def evaluate_layer_sweep(
    layer_states_train: dict[int | str, np.ndarray],
    y_train: np.ndarray,
    layer_states_val: dict[int | str, np.ndarray],
    y_val: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Sweeps through hidden states extracted from multiple transformer layers,
    fits a linear probe on train, and evaluates AUROC on val.

    Returns:
        results: list of dicts with layer, auroc, best_layer, and fitted_probes.
    """
    sweep_results = []
    fitted_probes = {}
    best_auroc = -1.0
    best_layer = None

    for layer_id, X_tr in layer_states_train.items():
        if layer_id not in layer_states_val:
            continue
        X_v = layer_states_val[layer_id]

        probe = HiddenStateProbe(C=C, random_state=random_state)
        probe.fit(X_tr, y_train)
        auroc = probe.score_auroc(X_v, y_val)

        fitted_probes[layer_id] = probe
        sweep_results.append({
            "layer": layer_id,
            "auroc": auroc,
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer_id

    return {
        "sweep": sweep_results,
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "probes": fitted_probes,
    }

