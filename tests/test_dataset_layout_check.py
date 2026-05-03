from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_json


class DatasetLayoutCheckTest(unittest.TestCase):
    def test_ready_fake_eurosat_layout_writes_preflight_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "eurosat"
            make_fake_dataset(dataset_root, num_classes=10, images_per_class=30)
            completed = run_preflight(dataset_root, root / "preflight")
            report = read_json(extract_path(completed.stdout, "preflight_report_path"))
            self.assertTrue(report["is_ready_for_split_generation"])
            self.assertEqual(report["dataset"], "eurosat")
            self.assertEqual(report["num_classes"], 10)
            self.assertEqual(report["num_images"], 300)
            self.assertEqual(report["supports_shots"]["16"], True)
            self.assertEqual(report["empty_classes"], [])
            self.assertEqual(report["duplicate_class_names"], [])
            self.assertEqual(report["invalid_class_names"], [])

    def test_insufficient_images_reports_unsupported_shots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "eurosat"
            make_fake_dataset(dataset_root, num_classes=10, images_per_class=4)
            completed = run_preflight(dataset_root, root / "preflight", check=False)
            self.assertEqual(completed.returncode, 2)
            report = read_json(extract_path(completed.stdout, "preflight_report_path"))
            self.assertFalse(report["is_ready_for_split_generation"])
            self.assertTrue(report["supports_shots"]["1"])
            self.assertTrue(report["supports_shots"]["2"])
            self.assertFalse(report["supports_shots"]["4"])
            self.assertFalse(report["supports_shots"]["16"])
            self.assertIn("class_00", report["shot_failures"]["4"])

    def test_non_image_files_are_reported_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "eurosat"
            make_fake_dataset(dataset_root, num_classes=10, images_per_class=30)
            (dataset_root / "class_00" / "notes.txt").write_text("not an image\n", encoding="utf-8")
            completed = run_preflight(dataset_root, root / "preflight")
            report = read_json(extract_path(completed.stdout, "preflight_report_path"))
            self.assertTrue(report["is_ready_for_split_generation"])
            self.assertTrue(report["has_non_image_files"])
            self.assertEqual(report["non_image_files"][".txt"], 1)
            self.assertTrue(report["warnings"])

    def test_invalid_class_names_block_split_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "eurosat"
            make_fake_dataset(dataset_root, num_classes=9, images_per_class=30)
            invalid_class = dataset_root / "bad@class"
            invalid_class.mkdir(parents=True)
            for index in range(30):
                (invalid_class / f"sample_{index:03d}.jpg").write_text("fake image\n", encoding="utf-8")
            completed = run_preflight(dataset_root, root / "preflight", check=False)
            self.assertEqual(completed.returncode, 2)
            report = read_json(extract_path(completed.stdout, "preflight_report_path"))
            self.assertFalse(report["is_ready_for_split_generation"])
            self.assertIn("bad@class", report["invalid_class_names"])


def make_fake_dataset(root: Path, num_classes: int, images_per_class: int) -> None:
    for class_idx in range(num_classes):
        class_dir = root / f"class_{class_idx:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_idx in range(images_per_class):
            (class_dir / f"sample_{image_idx:03d}.jpg").write_text("fake image\n", encoding="utf-8")


def run_preflight(dataset_root: Path, output_dir: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_dataset_layout.py",
            "--config",
            "configs/datasets/eurosat.yaml",
            "--dataset",
            "eurosat",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--execution-env",
            "local_wsl",
            "--run-mode",
            "local_validation",
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


if __name__ == "__main__":
    unittest.main()
