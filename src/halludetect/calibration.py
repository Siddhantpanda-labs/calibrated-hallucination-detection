"""
Calibration module for post-processing hallucination detector scores.
Implements Platt scaling, Isotonic regression, and Temperature scaling.
Calibration must be fitted exclusively on the held-out calibration split.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    """
    Platt Scaling: Fits a 1D logistic regression model:
    P(y=1 | s) = 1 / (1 + exp(A * s + B))
    where s is the uncalibrated score or log-odds.
    """
    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        # Avoid fitting on single-class data
        if len(np.unique(y)) > 1:
            self.model.fit(scores, y)
            self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(scores)[:, 1]


class IsotonicCalibrator:
    """
    Isotonic Regression: Non-parametric monotonic fitting.
    Transforms scores while preserving monotonic ranking.
    """
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) > 1:
            self.model.fit(scores, y)
            self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        scores = np.asarray(scores, dtype=float)
        probs = self.model.predict(scores)
        return np.clip(probs, 1e-4, 1.0 - 1e-4)


class TemperatureScaler:
    """
    Temperature Scaling: Learns a single positive scalar T:
    P(y=1) = sigma(z / T)
    where z is the uncalibrated logit: log(p / (1 - p)).
    """
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def _probs_to_logits(self, probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=float)
        logits = self._probs_to_logits(probs)

        # Loss: binary cross entropy as function of T
        def nll_loss(T: float) -> float:
            T_val = max(float(T[0]), 1e-3)
            scaled_logits = logits / T_val
            # Numerically stable log-loss
            loss = np.mean(
                np.maximum(scaled_logits, 0)
                - scaled_logits * y
                + np.log1p(np.exp(-np.abs(scaled_logits)))
            )
            return float(loss)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 20.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        logits = self._probs_to_logits(probs)
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))


class CalibratedDetector:
    """
    Wraps any base detector with a fitted calibrator.
    Base detector must provide .predict_proba() or .decision_function().
    """
    def __init__(self, base_detector, method: str = "platt"):
        self.base_detector = base_detector
        self.method = method.lower()
        if self.method == "platt":
            self.calibrator = PlattCalibrator()
        elif self.method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif self.method in ("temperature", "temp"):
            self.calibrator = TemperatureScaler()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit_base(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits base detector on training split."""
        self.base_detector.fit(X_train, y_train)
        return self

    def fit_calibration(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fits calibrator on held-out calibration split."""
        raw_probs = self.predict_raw(X_cal)
        self.calibrator.fit(raw_probs, y_cal)
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fits base model on train, then calibrator on cal."""
        self.fit_base(X_train, y_train)
        self.fit_calibration(X_cal, y_cal)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Gets uncalibrated scores or probabilities from base detector."""
        if hasattr(self.base_detector, "predict_proba"):
            probs = self.base_detector.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.base_detector, "decision_function"):
            scores = self.base_detector.decision_function(X)
            # convert scores via standard sigmoid
            return 1.0 / (1.0 + np.exp(-scores))
        elif hasattr(self.base_detector, "predict"):
            return self.base_detector.predict(X).astype(float)
        else:
            raise AttributeError("Base detector has neither predict_proba nor decision_function")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gets calibrated probabilities of hallucination."""
        raw_probs = self.predict_raw(X)
        return self.calibrator.predict_proba(raw_probs)

