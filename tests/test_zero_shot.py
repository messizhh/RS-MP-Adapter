from __future__ import annotations

import unittest

from src.baselines.zero_shot import ZeroShotClassifier
from src.features.feature_cache import make_fake_feature_cache


class ZeroShotTest(unittest.TestCase):
    def test_zero_shot_logits_shape(self) -> None:
        cache = make_fake_feature_cache(num_samples=12, num_classes=3, feature_dim=8)
        result = ZeroShotClassifier().evaluate_cache(cache)
        self.assertEqual(len(result.logits), 12)
        self.assertEqual(len(result.logits[0]), 3)
        self.assertTrue(result.used_fake_features)


if __name__ == "__main__":
    unittest.main()
