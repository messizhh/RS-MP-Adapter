from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.features.feature_cache import FeatureCache, load_feature_cache, make_fake_feature_cache, save_feature_cache


class FeatureCacheTest(unittest.TestCase):
    def test_fake_feature_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = make_fake_feature_cache(num_samples=9, num_classes=3, feature_dim=8, seed=1)
            path = save_feature_cache(cache, Path(temp_dir) / "features.pt")
            loaded = load_feature_cache(path)
            self.assertEqual(len(loaded.image_features), 9)
            self.assertEqual(len(loaded.image_features[0]), 8)
            self.assertEqual(len(loaded.image_labels), 9)
            self.assertEqual(len(loaded.image_paths), 9)
            self.assertEqual(loaded.split_name, "test")
            self.assertEqual(loaded.dataset, "eurosat")
            self.assertEqual(loaded.backbone, "fake_backbone")
            self.assertEqual(loaded.feature_dim, 8)
            self.assertEqual(loaded.class_to_idx, {"class_0": 0, "class_1": 1, "class_2": 2})
            self.assertIsNotNone(loaded.text_features)
            self.assertEqual(len(loaded.text_features), 3)
            self.assertEqual(len(loaded.text_features[0]), 8)
            self.assertFalse(loaded.metadata["is_real_feature_extraction"])

    def test_feature_cache_shape_validation_fails(self) -> None:
        cache = FeatureCache(
            image_features=[[0.0, 0.0], [0.0, 0.0]],
            image_labels=[0, 0, 0],
            image_paths=["a", "b"],
            split_name="test",
            class_to_idx={"a": 0, "b": 1},
            backbone="fake",
            dataset="fake",
            feature_dim=2,
            normalize_features=True,
            created_at="",
            source_script="tests/test_feature_cache.py",
        )
        with self.assertRaises(ValueError):
            cache.validate()

    def test_text_feature_class_count_validation_fails(self) -> None:
        cache = FeatureCache(
            image_features=[[0.0, 1.0], [1.0, 0.0]],
            image_labels=[0, 1],
            image_paths=["a", "b"],
            split_name="test",
            class_to_idx={"a": 0, "b": 1},
            text_features=[[0.0, 1.0]],
            backbone="fake",
            dataset="fake",
            feature_dim=2,
            normalize_features=True,
            created_at="",
            source_script="tests/test_feature_cache.py",
        )
        with self.assertRaisesRegex(ValueError, "class count"):
            cache.validate()

    def test_feature_dim_validation_fails(self) -> None:
        cache = FeatureCache(
            image_features=[[0.0, 1.0], [1.0, 0.0]],
            image_labels=[0, 1],
            image_paths=["a", "b"],
            split_name="test",
            class_to_idx={"a": 0, "b": 1},
            text_features=[[0.0, 1.0], [1.0, 0.0]],
            backbone="fake",
            dataset="fake",
            feature_dim=3,
            normalize_features=True,
            created_at="",
            source_script="tests/test_feature_cache.py",
        )
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            cache.validate()


if __name__ == "__main__":
    unittest.main()
