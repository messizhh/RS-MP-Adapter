from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.baselines.method_base import MethodBase


class DummyMethod(MethodBase):
    def fit(self, support_features, support_labels, val_features=None, val_labels=None):
        self.logits = [[1.0, 0.0]]
        self.class_names = ["a", "b"]
        return self

    def predict_logits(self, image_features):
        return self.logits


class MethodBaseTest(unittest.TestCase):
    def test_interface_evaluate_and_save_load(self) -> None:
        method = DummyMethod().fit([[0.0]], [0])
        metrics = method.evaluate([[0.0]], [0])
        self.assertEqual(metrics["top1_acc"], 1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = method.save(Path(temp_dir) / "method.pkl")
            loaded = DummyMethod.load(path)
            self.assertEqual(loaded.logits, [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
