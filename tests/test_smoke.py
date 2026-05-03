from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import logging

from src.logging.experiment_logger import create_unique_run_dir
from src.utils.io import read_json


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in smoke output: {stdout}")


class SmokeScriptTest(unittest.TestCase):
    def test_smoke_script_generates_non_paper_metadata_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_smoke_test.py",
                    "--dry-run",
                    "--run-mode",
                    "smoke_test",
                    "--execution-env",
                    "local_wsl",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(Path(temp_dir) / "smoke"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata_path = extract_path(completed.stdout, "metadata_path")
            metrics_path = extract_path(completed.stdout, "metrics_path")
            metadata = read_json(metadata_path)
            metrics = read_json(metrics_path)
            for required_file in ["config.yaml", "metadata.json", "metrics.json", "log.txt"]:
                self.assertTrue((metadata_path.parent / required_file).exists())
            metadata_required = {
                "run_id",
                "git_commit",
                "python_version",
                "pytorch_version",
                "cuda_version",
                "gpu_name",
                "host_name",
                "command",
                "config_path",
                "config_snapshot_path",
                "seed",
                "dataset",
                "shot",
                "backbone",
                "method",
                "execution_env",
                "run_mode",
                "is_paper_result",
                "device",
                "server_job_id",
                "split_path",
                "start_time",
                "end_time",
                "result_json_path",
                "log_path",
            }
            metrics_required = {
                "run_id",
                "method",
                "backbone",
                "dataset",
                "shot",
                "seed",
                "execution_env",
                "run_mode",
                "is_paper_result",
                "device",
                "top1_acc",
                "cache_entries",
                "trainable_params",
                "training_time_sec",
                "inference_time_sec",
                "images_per_second",
                "gpu_memory_mb",
                "config_path",
                "config_snapshot_path",
                "split_path",
                "result_json_path",
                "log_path",
                "start_time",
                "end_time",
                "fake_or_dry_run",
            }
            self.assertTrue(metadata_required.issubset(metadata))
            self.assertTrue(metrics_required.issubset(metrics))
            self.assertEqual(metadata["execution_env"], "local_wsl")
            self.assertEqual(metadata["run_mode"], "smoke_test")
            self.assertEqual(metadata["device"], "cpu")
            self.assertFalse(metadata["is_paper_result"])
            self.assertFalse(metrics["is_paper_result"])
            self.assertTrue(metrics["uses_fake_data"])
            self.assertTrue(metrics["uses_fake_features"])
            self.assertTrue(metrics["fake_or_dry_run"])
            self.assertFalse(metrics["is_real_evaluation"])

    def test_unique_run_dir_creation_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_id, first_dir = create_unique_run_dir(temp_dir, "eurosat", "fake", "zero_shot_clip", 1, 1)
            second_id, second_dir = create_unique_run_dir(temp_dir, "eurosat", "fake", "zero_shot_clip", 1, 1)
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_dir, second_dir)
            self.assertTrue(first_dir.exists())
            self.assertTrue(second_dir.exists())

    def test_standard_library_logging_is_not_shadowed(self) -> None:
        logging_path = Path(logging.__file__).as_posix()
        self.assertNotIn("/src/logging/", logging_path)


if __name__ == "__main__":
    unittest.main()
