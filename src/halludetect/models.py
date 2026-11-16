"""
Detector models for hallucination detection.
Includes unsupervised baselines, supervised tabular models, and combined multi-signal models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


class BaselineDetector(BaseEstimator, ClassifierMixin):
    """
    Unsupervised baseline detector based on a single heuristic signal.
    """
    def __init__(self, feature_name: str, higher_is_riskier: bool = True):
        self.feature_name = feature_name
        self.higher_is_riskier = higher_is_riskier
        self.min_val = 0.0
        self.max_val = 1.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None):
        vals = self._extract_vals(X)
        self.min_val = float(np.percentile(vals, 1))
        self.max_val = float(np.percentile(vals, 99))
        if self.max_val == self.min_val:
            self.max_val = self.min_val + 1.0
        return self

    def _extract_vals(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            vals = X[self.feature_name].values.astype(float)
        elif isinstance(X, np.ndarray):
            vals = X[:, 0].astype(float)
        else:
            vals = np.asarray(X, dtype=float)
        return vals

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        vals = self._extract_vals(X)
        # Normalize to [0, 1] approximately
        norm_scores = (vals - self.min_val) / (self.max_val - self.min_val)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)
        
        if not self.higher_is_riskier:
            # lower value was riskier (e.g. logprob or agreement rate)
            p1 = 1.0 - norm_scores
        else:
            p1 = norm_scores

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def create_baseline_logprob() -> BaselineDetector:
    """Raw mean token logprob baseline: lower logprob -> higher hallucination risk."""
    return BaselineDetector(feature_name="mean_logprob", higher_is_riskier=False)


def create_baseline_agreement() -> BaselineDetector:
    """Raw agreement rate baseline: lower agreement -> higher hallucination risk."""
    return BaselineDetector(feature_name="agreement_rate", higher_is_riskier=False)


def create_baseline_semantic_entropy() -> BaselineDetector:
    """Semantic entropy baseline: higher entropy -> higher hallucination risk."""
    return BaselineDetector(feature_name="semantic_entropy", higher_is_riskier=True)


class SupervisedDetector(BaseEstimator, ClassifierMixin):
    """
    Supervised classifier on extracted features.
    Supports 'logistic' (L2 regularized) and 'gbm' (Gradient Boosted Trees).
    """
    def __init__(
        self,
        model_type: str = "logistic",
        feature_names: Sequence[str] | None = None,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.feature_names = list(feature_names) if feature_names else None
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_type in ("gbm", "gradient_boosting"):
            self.model = HistGradientBoostingClassifier(
                random_state=self.random_state,
                max_iter=100,
                max_depth=4,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.is_fitted = False

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                # Use specified subset if present
                cols = [c for c in self.feature_names if c in X.columns]
                X_mat = X[cols].values.astype(float)
            else:
                X_mat = X.select_dtypes(include=[np.number]).values.astype(float)
        else:
            X_mat = np.asarray(X, dtype=float)
        return X_mat

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> SupervisedDetector:
        X_mat = self._prepare_X(X)
        y = np.asarray(y, dtype=int)
        
        if len(np.unique(y)) > 1:
            if self.model_type == "logistic":
                X_scaled = self.scaler.fit_transform(X_mat)
                self.model.fit(X_scaled, y)
            else:
                self.model.fit(X_mat, y)
            self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            n = len(X)
            return np.full((n, 2), 0.5)

        X_mat = self._prepare_X(X)
        if self.model_type == "logistic":
            X_scaled = self.scaler.transform(X_mat)
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X_mat)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# Standard feature column groups
GRAYBOX_FEATURES = [
    "mean_logprob",
    "min_logprob",
    "first_token_logprob",
    "mean_entropy",
    "max_entropy",
    "answer_length",
]

BLACKBOX_FEATURES = [
    "agreement_rate",
    "semantic_entropy",
    "num_semantic_clusters",
]

ALL_FEATURES = GRAYBOX_FEATURES + BLACKBOX_FEATURES

