"""
Unit tests for metrics and confidence intervals.
"""

import unittest
import numpy as np
from halludetect.metrics import (
    compute_auroc,
    compute_auprc,
    compute_brier_score,
    compute_ece,
    compute_risk_coverage,
    bootstrap_ci,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Perfect predictor
        self.y_prob_perfect = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])
        # Random predictor
        self.y_prob_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def test_auroc(self):
        auroc_perf = compute_auroc(self.y_true, self.y_prob_perfect)
        self.assertAlmostEqual(auroc_perf, 1.0, places=4)
        auroc_rand = compute_auroc(self.y_true, self.y_prob_random)
        self.assertAlmostEqual(auroc_rand, 0.5, places=4)

    def test_auprc(self):
        auprc_perf = compute_auprc(self.y_true, self.y_prob_perfect)
        self.assertAlmostEqual(auprc_perf, 1.0, places=4)

    def test_brier_score(self):
        brier = compute_brier_score(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(brier, 0.0, places=4)
        brier_wrong = compute_brier_score(np.array([0, 1]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(brier_wrong, 1.0, places=4)

    def test_ece(self):
        # When probabilities match empirical accuracy perfectly
        y_t = np.array([0, 0, 1, 1])
        y_p = np.array([0.0, 0.0, 1.0, 1.0])
        ece, mce, details = compute_ece(y_t, y_p, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=4)

        # Overconfident errors
        y_t_bad = np.array([0, 0, 0, 0])
        y_p_bad = np.array([0.9, 0.9, 0.9, 0.9])
        ece_bad, _, _ = compute_ece(y_t_bad, y_p_bad, n_bins=10)
        self.assertGreater(ece_bad, 0.5)

    def test_risk_coverage(self):
        coverages, risks, aurc = compute_risk_coverage(self.y_true, self.y_prob_perfect)
        self.assertEqual(len(coverages), len(risks))
        # At coverage 0.5 (first 4 items), risk is 0
        idx_half = np.searchsorted(coverages, 0.5)
        self.assertEqual(risks[idx_half], 0.0)
        # At coverage 1.0, risk is overall error rate (4/8 = 0.5)
        self.assertAlmostEqual(risks[-1], 0.5, places=4)

    def test_bootstrap_ci(self):
        ci = bootstrap_ci(self.y_true, self.y_prob_perfect, metric_name="auroc", n_bootstraps=50)
        self.assertIn("point", ci)
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)
        self.assertTrue(ci["ci_lower"] <= ci["point"] <= ci["ci_upper"])


if __name__ == "__main__":
    unittest.main()

