from __future__ import annotations

import unittest

from src.features.feature_cache import FeatureCache, load_feature_cache, make_fake_feature_cache, save_feature_cache


class FeatureCacheTest(unittest.TestCase):
    def test_fake_feature_cache_roundtrip(self) -> None:
        with self.subTest("roundtrip"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                cache = make_fake_feature_cache(num_samples=9, num_classes=3, feature_dim=8, seed=1)
                path = save_feature_cache(cache, Path(temp_dir) / "features.pt")
                loaded = load_feature_cache(path)
                self.assertEqual(len(loaded.image_features), 9)
                self.assertEqual(len(loaded.image_features[0]), 8)
                self.assertEqual(len(loaded.labels), 9)
                self.assertIsNotNone(loaded.text_features)
                self.assertEqual(len(loaded.text_features), 3)
                self.assertEqual(len(loaded.text_features[0]), 8)
                self.assertFalse(loaded.metadata["is_real_feature_extraction"])

    def test_feature_cache_shape_validation_fails(self) -> None:
        cache = FeatureCache(
            image_features=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            labels=[0, 0, 0],
            class_names=["a", "b"],
            metadata={},
        )
        with self.assertRaises(ValueError):
            cache.validate()


if __name__ == "__main__":
    unittest.main()
