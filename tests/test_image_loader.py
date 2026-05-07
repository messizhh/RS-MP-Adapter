from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.features.image_loader import (
    image_loading_safety_metadata,
    load_split_samples,
    resolve_relative_image_path,
)


class ImageLoaderTest(unittest.TestCase):
    def test_relative_paths_resolve_under_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            write_split(split_path, ["class_a/sample_0.jpg"])

            samples = load_split_samples(split_path, root, sections=["support"])

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].image_path, root / "class_a" / "sample_0.jpg")
            self.assertEqual(samples[0].relative_path, "class_a/sample_0.jpg")
            self.assertEqual(samples[0].label, 0)
            self.assertEqual(samples[0].class_name, "class_a")
            self.assertEqual(samples[0].section, "support")

    def test_absolute_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            absolute_path = str(root / "class_a" / "sample_0.jpg")

            with self.assertRaisesRegex(ValueError, "absolute"):
                resolve_relative_image_path(root, absolute_path)

    def test_paths_cannot_escape_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"

            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_relative_image_path(root, "../outside.jpg")

    def test_max_samples_limits_total_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            write_split(split_path, ["class_a/sample_0.jpg", "class_a/sample_1.jpg", "class_a/sample_2.jpg"])

            samples = load_split_samples(split_path, root, sections=["support"], max_samples=2)

            self.assertEqual([sample.relative_path for sample in samples], ["class_a/sample_0.jpg", "class_a/sample_1.jpg"])

    def test_loader_safety_metadata_marks_no_model_or_features(self) -> None:
        safety = image_loading_safety_metadata()

        self.assertFalse(safety["is_paper_result"])
        self.assertFalse(safety["loads_model"])
        self.assertFalse(safety["extracts_features"])
        self.assertFalse(safety["trains_model"])
        self.assertFalse(safety["evaluates_model"])


def write_split(path: Path, sample_paths: list[str]) -> None:
    samples = [{"path": sample_path, "label": 0, "class_name": "class_a"} for sample_path in sample_paths]
    data = {
        "dataset": "fake_dataset",
        "seed": 1,
        "shot": 1,
        "train": [],
        "val": [],
        "test": [],
        "support": samples,
        "class_to_idx": {"class_a": 0},
        "created_at": "2026-05-07T00:00:00+00:00",
        "source_script": "tests/test_image_loader.py",
    }
    path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
