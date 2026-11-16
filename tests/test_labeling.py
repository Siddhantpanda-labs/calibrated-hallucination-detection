"""
Unit tests for the labeling and normalization module.
Compatible with standard library unittest and pytest.
"""

import unittest
from halludetect.labeling import (
    normalize_text,
    get_tokens,
    is_exact_match,
    is_token_containment_match,
    label_prediction,
    label_truthfulqa_prediction,
    audit_labeling,
)


class TestLabeling(unittest.TestCase):
    def test_normalize_text(self):
        # Lowercase & punctuation
        self.assertEqual(normalize_text("Hello, World!"), "hello world")
        # Articles removal
        self.assertEqual(normalize_text("The Eiffel Tower"), "eiffel tower")
        self.assertEqual(normalize_text("An apple a day"), "apple day")
        # Whitespace stripping
        self.assertEqual(normalize_text("   lots   of    spaces   "), "lots of spaces")
        # Empty and None
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

    def test_is_exact_match(self):
        golds = ["William Shakespeare", "Shakespeare"]
        self.assertTrue(is_exact_match("william shakespeare", golds))
        self.assertTrue(is_exact_match("Shakespeare!", golds))
        self.assertFalse(is_exact_match("Francis Bacon", golds))
        self.assertFalse(is_exact_match("", golds))

    def test_is_token_containment_match(self):
        golds = ["Paris", "City of Paris"]
        self.assertTrue(is_token_containment_match("Paris, France", golds))
        self.assertTrue(is_token_containment_match("The Paris", golds))
        self.assertFalse(is_token_containment_match("London", golds))

        # Prevent substring inside word
        self.assertFalse(is_token_containment_match("car", ["carpet"]))
        self.assertFalse(is_token_containment_match("carpet", ["car"]))

        # Multi-token subset
        self.assertTrue(is_token_containment_match("United States", ["United States of America"]))

    def test_label_prediction(self):
        golds = ["Albert Einstein", "Einstein"]
        # 0 = correct, 1 = hallucinated
        self.assertEqual(label_prediction("Albert Einstein", golds), 0)
        self.assertEqual(label_prediction("Einstein", golds), 0)
        self.assertEqual(label_prediction("Isaac Newton", golds), 1)
        self.assertEqual(label_prediction("", golds), 1)

    def test_label_truthfulqa(self):
        correct = ["Earth is an oblate spheroid.", "Spherical"]
        incorrect = ["The Earth is flat.", "Flat"]

        self.assertEqual(label_truthfulqa_prediction("The earth is spherical", correct, incorrect), 0)
        self.assertEqual(label_truthfulqa_prediction("The Earth is completely flat", correct, incorrect), 1)
        self.assertEqual(label_truthfulqa_prediction("Donuts are round", correct, incorrect), 1)

    def test_audit_labeling(self):
        preds = ["Paris", "Mars", "Einstein"]
        golds = [["Paris"], ["Earth"], ["Albert Einstein"]]
        labels = [0, 1, 0]
        audit = audit_labeling(preds, golds, labels, sample_size=3)
        self.assertEqual(len(audit), 3)
        self.assertTrue(all("suspicious" in item for item in audit))


if __name__ == "__main__":
    unittest.main()
