from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.datasets.base_dataset import inspect_class_folder_dataset, list_class_folder_samples
from src.datasets.dataset_registry import get_dataset_descriptor
from src.utils.io import read_json
from tests.helpers.fake_datasets import create_fake_aid, create_fake_class_dataset, create_fake_eurosat, create_fake_nwpu


class DatasetInspectionTest(unittest.TestCase):
    def test_class_discovery_deterministic_and_hidden_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_fake_class_dataset(root, ["b", "a"], samples_per_class=2, include_hidden=True, include_unsupported=True)
            descriptor = get_dataset_descriptor("eurosat", root=root).with_options(expected_num_classes=None)
            samples, class_to_idx = list_class_folder_samples(descriptor)
            self.assertEqual(class_to_idx, {"a": 0, "b": 1})
            self.assertEqual(len(samples), 4)
            self.assertTrue(all(".hidden" not in sample.path for sample in samples))
            report = inspect_class_folder_dataset(descriptor)
            self.assertEqual(report["unsupported_extensions"], {".txt": 2})

    def test_empty_class_is_invalid_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_fake_class_dataset(root, ["a", "b"], samples_per_class=2, empty_class="b")
            descriptor = get_dataset_descriptor("eurosat", root=root).with_options(expected_num_classes=None)
            report = inspect_class_folder_dataset(descriptor)
            self.assertFalse(report["is_valid"])
            self.assertIn("no supported images", "; ".join(report["critical_errors"]))
            with self.assertRaisesRegex(ValueError, "no supported images"):
                list_class_folder_samples(descriptor)

    def test_dataset_layout_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            create_fake_eurosat(base / "eurosat")
            create_fake_aid(base / "aid")
            create_fake_nwpu(base / "nwpu")
            for name, root in [("eurosat", base / "eurosat"), ("aid", base / "aid"), ("nwpu_resisc45", base / "nwpu")]:
                descriptor = get_dataset_descriptor(name, root=root).with_options(expected_num_classes=None)
                samples, class_to_idx = list_class_folder_samples(descriptor)
                self.assertEqual(len(class_to_idx), 3)
                self.assertEqual(len(samples), 60)

    def test_inspect_dataset_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_fake_class_dataset(root / "data", samples_per_class=3, include_unsupported=True)
            output_dir = root / "reports"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/inspect_dataset.py",
                    "--dataset",
                    "eurosat",
                    "--dataset-root",
                    str(root / "data"),
                    "--output-dir",
                    str(output_dir),
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                    "--write-report",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report_path = extract_path(completed.stdout, "report_path")
            summary_path = extract_path(completed.stdout, "class_summary_path")
            report = read_json(report_path)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["num_classes"], 3)
            self.assertTrue(summary_path.exists())


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


if __name__ == "__main__":
    unittest.main()
