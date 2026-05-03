from __future__ import annotations

import unittest

from src.eval.evaluator import evaluate_logits
from src.eval.metrics import confusion_matrix, mean_std_over_seeds, per_class_accuracy, top1_accuracy


class MetricsTest(unittest.TestCase):
    def test_top1_accuracy_and_evaluator(self) -> None:
        logits = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
        labels = [0, 1, 1]
        self.assertAlmostEqual(top1_accuracy(logits, labels), 2 / 3)
        metrics = evaluate_logits(logits, labels, class_names=["a", "b"])
        self.assertAlmostEqual(metrics["top1_acc"], 2 / 3)
        self.assertEqual(metrics["num_samples"], 3)
        self.assertEqual(metrics["num_classes"], 2)
        self.assertEqual(metrics["predictions"], [0, 1, 0])
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 1]])

    def test_per_class_accuracy(self) -> None:
        rows = per_class_accuracy([0, 1, 0], [0, 1, 1], class_names=["a", "b"])
        self.assertEqual(rows[0]["num_samples"], 1)
        self.assertEqual(rows[0]["num_correct"], 1)
        self.assertEqual(rows[1]["num_samples"], 2)
        self.assertEqual(rows[1]["num_correct"], 1)
        self.assertAlmostEqual(rows[1]["accuracy"], 0.5)

    def test_confusion_matrix(self) -> None:
        self.assertEqual(confusion_matrix([0, 1, 0], [0, 1, 1], 2), [[1, 0], [1, 1]])

    def test_mean_std_over_seeds(self) -> None:
        result = mean_std_over_seeds([1.0, 2.0, 3.0])
        self.assertEqual(result["num_seeds"], 3)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], 1.0)
        empty = mean_std_over_seeds([])
        self.assertIsNone(empty["mean"])
        self.assertIsNone(empty["std"])
        self.assertEqual(empty["num_seeds"], 0)


if __name__ == "__main__":
    unittest.main()
