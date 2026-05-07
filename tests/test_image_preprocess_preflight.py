from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.utils.io import read_json


class ImagePreprocessPreflightTest(unittest.TestCase):
    def test_cli_success_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, split_path, config_path, output_dir = make_inputs(Path(temp_dir), ["class_a/sample_0.png"])
            make_image(root / "class_a" / "sample_0.png", size=(4, 3))

            completed = run_preflight(root, split_path, config_path, output_dir)

            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["dataset"], "fake_dataset")
            self.assertEqual(report["image_size"], 6)
            self.assertEqual(report["num_checked"], 1)
            self.assertEqual(report["num_failed"], 0)
            self.assertEqual(report["image_summaries"][0]["width"], 6)
            self.assertEqual(report["image_summaries"][0]["height"], 6)
            self.assertEqual(report["image_summaries"][0]["mode"], "RGB")

    def test_max_samples_per_section_limits_each_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            split_path = Path(temp_dir) / "split.json"
            config_path = Path(temp_dir) / "backbone.yaml"
            output_dir = Path(temp_dir) / "reports"
            write_config(config_path, resize=6)
            samples = ["class_a/train_0.png", "class_a/train_1.png", "class_a/train_2.png"]
            for sample_path in samples:
                make_image(root / sample_path)
            write_split(split_path, train=samples, val=["class_a/val_0.png"])
            make_image(root / "class_a" / "val_0.png")

            completed = run_preflight(root, split_path, config_path, output_dir, sections=["train", "val"])

            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["num_checked"], 3)
            self.assertEqual(len(report["image_summaries"]), 3)
            self.assertEqual(
                [summary["sample_path"] for summary in report["image_summaries"]],
                ["class_a/train_0.png", "class_a/train_1.png", "class_a/val_0.png"],
            )

    def test_absolute_path_fails_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, split_path, config_path, output_dir = make_inputs(Path(temp_dir), [])
            absolute_path = str(root / "class_a" / "sample_0.png")
            write_split(split_path, train=[absolute_path])

            completed = run_preflight(root, split_path, config_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["num_failed"], 1)
            self.assertIn("absolute", report["failures"][0]["error"])

    def test_missing_image_fails_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, split_path, config_path, output_dir = make_inputs(Path(temp_dir), ["class_a/missing.png"])

            completed = run_preflight(root, split_path, config_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertEqual(report["num_checked"], 1)
            self.assertEqual(report["num_failed"], 1)
            self.assertEqual(report["failures"][0]["sample_path"], "class_a/missing.png")

    def test_report_safety_flags_are_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, split_path, config_path, output_dir = make_inputs(Path(temp_dir), ["class_a/sample_0.png"])
            make_image(root / "class_a" / "sample_0.png")

            completed = run_preflight(root, split_path, config_path, output_dir)
            report = read_json(extract_path(completed.stdout))

            self.assertFalse(report["is_paper_result"])
            self.assertTrue(report["reads_image_pixels"])
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["image_summaries"][0]["loads_model"])
            self.assertFalse(report["image_summaries"][0]["extracts_features"])

    def test_output_path_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, split_path, config_path, output_dir = make_inputs(Path(temp_dir), ["class_a/sample_0.png"])
            make_image(root / "class_a" / "sample_0.png")

            first = run_preflight(root, split_path, config_path, output_dir)
            second = run_preflight(root, split_path, config_path, output_dir)

            first_path = extract_path(first.stdout)
            second_path = extract_path(second.stdout)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())


def make_inputs(base: Path, train_samples: list[str]) -> tuple[Path, Path, Path, Path]:
    root = base / "dataset"
    split_path = base / "split.json"
    config_path = base / "backbone.yaml"
    output_dir = base / "reports"
    write_split(split_path, train=train_samples)
    write_config(config_path, resize=6)
    return root, split_path, config_path, output_dir


def make_image(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path, format="PNG")


def write_config(path: Path, resize: int) -> None:
    data = {
        "backbone": {
            "name": "fake_backbone",
            "family": "fake",
            "weights": None,
            "allow_download": False,
            "image_size": 224,
            "preprocess": {"resize": resize, "center_crop": resize, "normalize": True},
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def write_split(path: Path, train: list[str], val: list[str] | None = None) -> None:
    data = {
        "dataset": "fake_dataset",
        "seed": 1,
        "shot": 1,
        "train": [make_sample(sample_path) for sample_path in train],
        "val": [make_sample(sample_path) for sample_path in val or []],
        "test": [],
        "support": [],
        "class_to_idx": {"class_a": 0},
        "created_at": "2026-05-07T00:00:00+00:00",
        "source_script": "tests/test_image_preprocess_preflight.py",
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def make_sample(sample_path: str) -> dict[str, object]:
    return {"path": sample_path, "label": 0, "class_name": "class_a"}


def run_preflight(
    root: Path,
    split_path: Path,
    config_path: Path,
    output_dir: Path,
    sections: list[str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_image_preprocess_preflight.py",
            "--dataset",
            "fake_dataset",
            "--dataset-root",
            str(root),
            "--split",
            str(split_path),
            "--backbone-config",
            str(config_path),
            "--sections",
            *(sections or ["train", "val", "test", "support"]),
            "--max-samples-per-section",
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
        if line.startswith("image_preprocess_report_path="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing image_preprocess_report_path in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
