from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
import tempfile
import subprocess
import sys

from src.datasets.base_dataset import DatasetSample, list_class_folder_samples
from src.datasets.dataset_registry import get_dataset_descriptor
from src.datasets.split_generator import build_support_set, generate_split_files, make_split_payload, split_samples_by_class
from src.utils.io import read_json, write_json
from tests.helpers.fake_datasets import create_fake_class_dataset


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
            dataset_root="/fake",
            split_policy="class_stratified_random",
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            image_extensions=[".jpg"],
            execution_env="local_wsl",
            run_mode="smoke_test",
            is_paper_result=False,
        )
        required = {
            "dataset",
            "seed",
            "shot",
            "train",
            "val",
            "test",
            "support",
            "class_to_idx",
            "created_at",
            "source_script",
            "dataset_root",
            "split_policy",
            "split_ratios",
            "image_extensions",
            "num_classes",
            "num_train",
            "num_val",
            "num_test",
            "num_support",
            "is_paper_result",
        }
        self.assertTrue(required.issubset(payload))

    def test_split_write_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shot_1_seed1.json"
            write_json(path, {"dataset": "eurosat"}, overwrite=False)
            with self.assertRaises(FileExistsError):
                write_json(path, {"dataset": "eurosat"}, overwrite=False)
            write_json(path, {"dataset": "eurosat", "seed": 1}, overwrite=True)

    def test_generate_split_files_overwrite_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            output = Path(temp_dir) / "splits"
            create_fake_class_dataset(root, samples_per_class=20)
            descriptor = get_dataset_descriptor("eurosat", root=root).with_options(expected_num_classes=None)
            written = generate_split_files(
                dataset="eurosat",
                root=root,
                output_dir=output,
                shots=[1],
                seeds=[1],
                descriptor=descriptor,
                execution_env="local_wsl",
                run_mode="smoke_test",
            )
            self.assertEqual(len(written), 2)
            split = read_json(output / "shot_1_seed1.json")
            self.assertEqual(split["num_support"], 3)
            self.assertFalse(split["is_paper_result"])
            with self.assertRaises(FileExistsError):
                generate_split_files("eurosat", root, output, shots=[1], seeds=[1], descriptor=descriptor)
            generate_split_files("eurosat", root, output, shots=[1], seeds=[1], descriptor=descriptor, overwrite=True)

    def test_generate_splits_cli_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            output = Path(temp_dir) / "splits"
            create_fake_class_dataset(root, class_names=[f"class_{idx}" for idx in range(10)], samples_per_class=20)
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_splits.py",
                    "--config",
                    "configs/datasets/eurosat.yaml",
                    "--dataset",
                    "eurosat",
                    "--dataset-root",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--shots",
                    "1",
                    "--seeds",
                    "1",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("dry_run_path=", completed.stdout)
            self.assertFalse((output / "shot_1_seed1.json").exists())

    def test_max_samples_per_class_and_min_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            create_fake_class_dataset(root, samples_per_class=5)
            descriptor = get_dataset_descriptor("eurosat", root=root).with_options(expected_num_classes=None)
            samples, _ = list_class_folder_samples(descriptor, max_samples_per_class=2)
            self.assertEqual(len(samples), 6)
            with self.assertRaisesRegex(ValueError, "fewer than min_images_per_class"):
                list_class_folder_samples(descriptor, min_samples_per_class=6)


if __name__ == "__main__":
    unittest.main()
