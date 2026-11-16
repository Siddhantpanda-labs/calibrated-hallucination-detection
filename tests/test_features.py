"""
Unit tests for feature extraction modules (gray-box, black-box, white-box probe).
"""

import unittest
import numpy as np
from halludetect.features.grey import compute_entropy_from_logits, extract_graybox_features
from halludetect.features.consistency import (
    compute_agreement_rate,
    cluster_by_string,
    compute_semantic_entropy,
    extract_blackbox_features,
)
from halludetect.features.probe import HiddenStateProbe, evaluate_layer_sweep


class TestFeatures(unittest.TestCase):
    def test_compute_entropy_from_logits(self):
        # Uniform distribution over 4 tokens -> entropy = ln(4) = ~1.3863
        uniform_logits = np.array([2.0, 2.0, 2.0, 2.0])
        ent = compute_entropy_from_logits(uniform_logits)
        self.assertAlmostEqual(ent, np.log(4), places=3)

        # Peaked distribution -> entropy close to 0
        peaked_logits = np.array([100.0, 0.0, 0.0, 0.0])
        ent_peak = compute_entropy_from_logits(peaked_logits)
        self.assertAlmostEqual(ent_peak, 0.0, places=2)

    def test_extract_graybox_features(self):
        logprobs = [-0.1, -0.5, -1.2, -0.2]
        entropies = [0.2, 0.8, 1.5, 0.3]
        feats = extract_graybox_features(logprobs, entropies)
        self.assertAlmostEqual(feats["mean_logprob"], np.mean(logprobs), places=4)
        self.assertAlmostEqual(feats["min_logprob"], -1.2, places=4)
        self.assertAlmostEqual(feats["first_token_logprob"], -0.1, places=4)
        self.assertAlmostEqual(feats["mean_entropy"], np.mean(entropies), places=4)
        self.assertAlmostEqual(feats["max_entropy"], 1.5, places=4)
        self.assertEqual(feats["answer_length"], 4.0)

    def test_extract_blackbox_features(self):
        greedy = "Paris"
        samples = ["Paris", "City of Paris", "London", "Paris"]
        feats = extract_blackbox_features(greedy, samples)
        # 3 out of 4 agree with Paris
        self.assertAlmostEqual(feats["agreement_rate"], 0.75, places=4)
        self.assertGreater(feats["semantic_entropy"], 0.0)
        self.assertEqual(feats["num_semantic_clusters"], 2.0)

    def test_hidden_state_probe_and_layer_sweep(self):
        np.random.seed(42)
        n_samples = 100
        dim = 16
        # Layer 1 has random noise; Layer 2 has signal
        y_train = np.random.binomial(1, 0.5, size=n_samples)
        y_val = np.random.binomial(1, 0.5, size=n_samples)

        X_train_l1 = np.random.randn(n_samples, dim)
        X_val_l1 = np.random.randn(n_samples, dim)

        X_train_l2 = np.random.randn(n_samples, dim) + 1.5 * y_train[:, None]
        X_val_l2 = np.random.randn(n_samples, dim) + 1.5 * y_val[:, None]

        layer_states_train = {1: X_train_l1, 2: X_train_l2}
        layer_states_val = {1: X_val_l1, 2: X_val_l2}

        sweep = evaluate_layer_sweep(
            layer_states_train, y_train, layer_states_val, y_val, random_state=42
        )
        self.assertEqual(sweep["best_layer"], 2)
        self.assertGreater(sweep["best_auroc"], 0.7)


if __name__ == "__main__":
    unittest.main()

