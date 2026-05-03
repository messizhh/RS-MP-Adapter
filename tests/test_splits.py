from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
import tempfile

from src.datasets.base_dataset import DatasetSample
from src.datasets.split_generator import build_support_set, make_split_payload, split_samples_by_class
from src.utils.io import write_json


def make_samples(classes: int = 3, samples_per_class: int = 20) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for label in range(classes):
        for idx in range(samples_per_class):
            samples.append(DatasetSample(path=f"/fake/class_{label}/{idx}.jpg", label=label, class_name=f"class_{label}"))
    return samples


def paths(samples: list[DatasetSample]) -> list[str]:
    return [sample.path for sample in samples]


class SplitGenerationTest(unittest.TestCase):
    def test_split_reproducibility_same_seed(self) -> None:
        samples = make_samples()
        first = split_samples_by_class(samples, seed=1)
        second = split_samples_by_class(samples, seed=1)
        self.assertEqual([paths(part) for part in first], [paths(part) for part in second])

    def test_split_different_seed_may_differ(self) -> None:
        samples = make_samples()
        first = split_samples_by_class(samples, seed=1)
        second = split_samples_by_class(samples, seed=2)
        self.assertNotEqual([paths(part) for part in first], [paths(part) for part in second])

    def test_support_has_requested_shot_per_class(self) -> None:
        train, _, _ = split_samples_by_class(make_samples(), seed=1)
        support = build_support_set(train, shot=4, seed=1)
        counts = Counter(sample.label for sample in support)
        self.assertEqual(counts, {0: 4, 1: 4, 2: 4})

    def test_insufficient_support_samples_raise_clear_error(self) -> None:
        train, _, _ = split_samples_by_class(make_samples(samples_per_class=2), seed=1)
        with self.assertRaisesRegex(ValueError, "fewer than shot=4"):
            build_support_set(train, shot=4, seed=1)

    def test_split_payload_contains_required_fields(self) -> None:
        train, val, test = split_samples_by_class(make_samples(), seed=1)
        support = build_support_set(train, shot=1, seed=1)
        payload = make_split_payload(
            dataset="eurosat",
            seed=1,
            shot=1,
            train=train,
            val=val,
            test=test,
            support=support,
            class_to_idx={"class_0": 0, "class_1": 1, "class_2": 2},
            source_script="tests/test_splits.py",
        )
        required = {"dataset", "seed", "shot", "train", "val", "test", "support", "class_to_idx", "created_at", "source_script"}
        self.assertTrue(required.issubset(payload))

    def test_split_write_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shot_1_seed1.json"
            write_json(path, {"dataset": "eurosat"}, overwrite=False)
            with self.assertRaises(FileExistsError):
                write_json(path, {"dataset": "eurosat"}, overwrite=False)
            write_json(path, {"dataset": "eurosat", "seed": 1}, overwrite=True)


if __name__ == "__main__":
    unittest.main()
