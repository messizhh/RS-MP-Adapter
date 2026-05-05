from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.utils.io import read_json


class ImageOpenPreflightTest(unittest.TestCase):
    def test_cli_success_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            output_dir = Path(temp_dir) / "reports"
            make_image(root / "class_a" / "sample.jpg")
            write_split(split_path, "class_a/sample.jpg")

            completed = run_preflight(root, split_path, output_dir)

            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["dataset"], "fake_dataset")
            self.assertEqual(report["num_checked"], 1)
            self.assertEqual(report["num_failed"], 0)
            self.assertEqual(report["image_summaries"][0]["width"], 4)
            self.assertEqual(report["image_summaries"][0]["height"], 3)
            self.assertEqual(report["image_summaries"][0]["mode"], "RGB")
            self.assertEqual(report["image_summaries"][0]["format"], "PNG")

    def test_absolute_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            output_dir = Path(temp_dir) / "reports"
            make_image(root / "class_a" / "sample.jpg")
            write_split(split_path, str(root / "class_a" / "sample.jpg"))

            completed = run_preflight(root, split_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["num_failed"], 1)
            self.assertIn("absolute", report["failures"][0]["error"])

    def test_missing_image_fails_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            output_dir = Path(temp_dir) / "reports"
            write_split(split_path, "class_a/missing.jpg")

            completed = run_preflight(root, split_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["num_checked"], 1)
            self.assertEqual(report["num_failed"], 1)
            self.assertEqual(report["failures"][0]["sample_path"], "class_a/missing.jpg")

    def test_report_safety_flags_are_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            output_dir = Path(temp_dir) / "reports"
            make_image(root / "class_a" / "sample.jpg")
            write_split(split_path, "class_a/sample.jpg")

            completed = run_preflight(root, split_path, output_dir)
            report = read_json(extract_path(completed.stdout))

            self.assertFalse(report["is_paper_result"])
            self.assertTrue(report["reads_image_pixels"])
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])

    def test_output_path_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            output_dir = Path(temp_dir) / "reports"
            make_image(root / "class_a" / "sample.jpg")
            write_split(split_path, "class_a/sample.jpg")

            first = run_preflight(root, split_path, output_dir)
            second = run_preflight(root, split_path, output_dir)

            first_path = extract_path(first.stdout)
            second_path = extract_path(second.stdout)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    image.save(path, format="PNG")


def write_split(path: Path, sample_path: str) -> None:
    sample = {"path": sample_path, "label": 0, "class_name": "class_a"}
    data = {
        "dataset": "fake_dataset",
        "seed": 1,
        "shot": 1,
        "train": [sample],
        "val": [],
        "test": [],
        "support": [],
        "class_to_idx": {"class_a": 0},
        "created_at": "2026-05-05T00:00:00+00:00",
        "source_script": "tests/test_image_open_preflight.py",
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def run_preflight(root: Path, split_path: Path, output_dir: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_image_open_preflight.py",
            "--dataset",
            "fake_dataset",
            "--dataset-root",
            str(root),
            "--split",
            str(split_path),
            "--sections",
            "train",
            "val",
            "test",
            "support",
            "--max-samples",
            "2",
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


def extract_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("image_open_report_path="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing image_open_report_path in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
