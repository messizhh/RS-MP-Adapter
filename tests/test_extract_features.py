from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.features.feature_cache import load_feature_cache
from src.features.extract_features import run_dry_run_feature_extraction
from src.utils.io import read_json


class ExtractFeaturesTest(unittest.TestCase):
    def test_dry_run_extraction_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_dry_run_feature_extraction(
                dataset="eurosat",
                backbone_name="fake_backbone",
                backbone_config={"backbone": {"name": "fake_backbone", "family": "fake", "feature_dim": 8}},
                output_dir=temp_dir,
                split_path=None,
                max_samples=12,
                batch_size=4,
                device="cpu",
                execution_env="local_wsl",
                run_mode="smoke_test",
                prompt_templates=["a satellite photo of {}.", "a remote sensing image of {}."],
            )
            cache = load_feature_cache(result["cache_path"])
            self.assertEqual(len(cache.image_features), 12)
            self.assertEqual(len(cache.image_features[0]), 8)
            self.assertEqual(len(cache.text_features), 3)
            self.assertEqual(len(cache.text_prompts), 6)
            self.assertFalse(cache.metadata["is_paper_result"])
            summary = read_json(result["summary_path"])
            self.assertFalse(summary["is_paper_result"])

    def test_extract_features_cli_and_validation_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "features"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/extract_features.py",
                    "--dataset",
                    "eurosat",
                    "--backbone",
                    "fake_backbone",
                    "--dry-run",
                    "--max-samples",
                    "12",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cache_path = extract_path(completed.stdout, "feature_cache_path")
            validation = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_feature_cache.py",
                    "--feature-cache",
                    str(cache_path),
                    "--output-dir",
                    str(Path(temp_dir) / "validation"),
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report_path = extract_path(validation.stdout, "validation_report_path")
            report = read_json(report_path)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["num_images"], 12)
            self.assertEqual(report["feature_dim"], 8)
            self.assertTrue(report["uses_fake_features"])

    def test_extraction_unique_paths_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs = dict(
                dataset="eurosat",
                backbone_name="fake_backbone",
                backbone_config={"backbone": {"name": "fake_backbone", "family": "fake", "feature_dim": 8}},
                output_dir=temp_dir,
                split_path=None,
                max_samples=3,
                batch_size=2,
                device="cpu",
                execution_env="local_wsl",
                run_mode="smoke_test",
            )
            first = run_dry_run_feature_extraction(**kwargs)
            second = run_dry_run_feature_extraction(**kwargs)
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertTrue(first["cache_path"].exists())
            self.assertTrue(second["cache_path"].exists())


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


if __name__ == "__main__":
    unittest.main()
