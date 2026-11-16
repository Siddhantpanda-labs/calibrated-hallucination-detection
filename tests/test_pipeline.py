"""
End-to-end integration test for the full hallucination detection pipeline.
Validates that all components, features, models, calibrators, and plots run seamlessly together.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from halludetect.data import generate_mock_dataset, split_by_question
from halludetect.generate import MockGenerator
from halludetect.labeling import label_prediction
from halludetect.features.grey import extract_graybox_features
from halludetect.features.consistency import extract_blackbox_features
from halludetect.features.probe import HiddenStateProbe, evaluate_layer_sweep
from halludetect.models import (
    create_baseline_logprob,
    SupervisedDetector,
    GRAYBOX_FEATURES,
    BLACKBOX_FEATURES,
    ALL_FEATURES,
)
from halludetect.calibration import CalibratedDetector
from halludetect.metrics import compute_all_metrics, compute_ece
from halludetect.plots import (
    plot_reliability_diagram,
    plot_layer_sweep,
    plot_risk_coverage_curve,
)


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_flow(self):
        # 1. Dataset generation
        df_raw = generate_mock_dataset(num_samples=40, seed=42)
        self.assertEqual(len(df_raw), 40)

        # 2. Generation simulation
        generator = MockGenerator(hidden_dim=32, layers=["4", "8", "-1"], num_samples=3, seed=42)
        records = []
        for _, row in df_raw.iterrows():
            res = generator.generate(row["question_id"], row["question"], row["gold_answers"])
            lbl = label_prediction(res.greedy_text, row["gold_answers"])
            gray = extract_graybox_features(res.token_logprobs, res.token_entropies)
            black = extract_blackbox_features(res.greedy_text, res.sampled_answers)
            
            records.append({
                "question_id": res.question_id,
                "question": res.question,
                "greedy_text": res.greedy_text,
                "label": lbl,
                **gray,
                **black,
                "hidden_states": res.hidden_states,
            })

        df_processed = pd.DataFrame(records)
        self.assertEqual(len(df_processed), 40)

        # 3. Disjoint Question Split
        df_train, df_cal, df_test = split_by_question(
            df_processed, train_frac=0.6, cal_frac=0.2, test_frac=0.2, seed=42
        )
        self.assertGreater(len(df_train), 0)
        self.assertGreater(len(df_cal), 0)
        self.assertGreater(len(df_test), 0)

        # 4. Layer sweep
        layer_train = {
            l: np.array([r[l] for r in df_train["hidden_states"]]) for l in ["4", "8", "-1"]
        }
        layer_cal = {
            l: np.array([r[l] for r in df_cal["hidden_states"]]) for l in ["4", "8", "-1"]
        }
        sweep = evaluate_layer_sweep(layer_train, df_train["label"].values, layer_cal, df_cal["label"].values)
        self.assertIn("best_layer", sweep)
        best_layer = sweep["best_layer"]

        # 5. Model training & Calibration
        model = SupervisedDetector("logistic", feature_names=ALL_FEATURES, random_state=42)
        calibrated = CalibratedDetector(model, method="platt")
        calibrated.fit(
            df_train, df_train["label"].values, df_cal, df_cal["label"].values
        )

        test_probs = calibrated.predict_proba(df_test)
        self.assertEqual(len(test_probs), len(df_test))
        self.assertTrue(np.all((test_probs >= 0.0) & (test_probs <= 1.0)))

        # 6. Metrics
        metrics = compute_all_metrics(df_test["label"].values, test_probs, n_bootstraps=10)
        self.assertIn("auroc", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)

        # 7. Plots
        fig_rel_path = os.path.join(self.temp_dir, "test_rel.png")
        plot_reliability_diagram(
            df_test["label"].values,
            {"Combined": test_probs},
            save_path=fig_rel_path,
        )
        self.assertTrue(os.path.exists(fig_rel_path))


if __name__ == "__main__":
    unittest.main()

