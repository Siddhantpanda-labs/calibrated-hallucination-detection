"""
Unit tests for calibration methods (Platt, Isotonic, Temperature Scaling).
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from halludetect.calibration import (
    PlattCalibrator,
    IsotonicCalibrator,
    TemperatureScaler,
    CalibratedDetector,
)
from halludetect.metrics import compute_ece


class TestCalibration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate synthetic uncalibrated scores: overconfident
        n = 500
        self.y_true = np.random.binomial(1, 0.4, size=n)
        # Raw scores that correlate with truth but have bad calibration (shifted and scaled)
        noise = np.random.normal(0, 0.3, size=n)
        self.raw_scores = np.clip(0.3 * self.y_true + 0.6 + noise, 0.01, 0.99)

        # Split into cal and test
        self.cal_scores = self.raw_scores[:250]
        self.cal_y = self.y_true[:250]
        self.test_scores = self.raw_scores[250:]
        self.test_y = self.y_true[250:]

    def test_platt_scaling(self):
        calibrator = PlattCalibrator().fit(self.cal_scores, self.cal_y)
        cal_probs = calibrator.predict_proba(self.test_scores)
        ece_before, _, _ = compute_ece(self.test_y, self.test_scores)
        ece_after, _, _ = compute_ece(self.test_y, cal_probs)
        self.assertTrue(cal_probs.min() >= 0.0 and cal_probs.max() <= 1.0)
        self.assertLess(ece_after, ece_before)

    def test_isotonic_regression(self):
        calibrator = IsotonicCalibrator().fit(self.cal_scores, self.cal_y)
        cal_probs = calibrator.predict_proba(self.test_scores)
        ece_before, _, _ = compute_ece(self.test_y, self.test_scores)
        ece_after, _, _ = compute_ece(self.test_y, cal_probs)
        self.assertTrue(cal_probs.min() >= 0.0 and cal_probs.max() <= 1.0)
        self.assertLess(ece_after, ece_before)

    def test_temperature_scaling(self):
        scaler = TemperatureScaler().fit(self.cal_scores, self.cal_y)
        cal_probs = scaler.predict_proba(self.test_scores)
        self.assertTrue(scaler.is_fitted)
        self.assertGreater(scaler.temperature, 0.0)
        self.assertTrue(cal_probs.min() >= 0.0 and cal_probs.max() <= 1.0)

    def test_calibrated_detector_wrapper(self):
        X_train = np.random.randn(200, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_cal = np.random.randn(100, 4)
        y_cal = (X_cal[:, 0] > 0).astype(int)
        X_test = np.random.randn(100, 4)

        base = LogisticRegression()
        detector = CalibratedDetector(base, method="platt")
        detector.fit(X_train, y_train, X_cal, y_cal)

        probs = detector.predict_proba(X_test)
        self.assertEqual(len(probs), 100)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


if __name__ == "__main__":
    unittest.main()

