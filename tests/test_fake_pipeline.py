from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_json


class FakePipelineTest(unittest.TestCase):
    def test_fake_pipeline_completes_and_exports_zero_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "fake_pipeline"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_fake_pipeline.py",
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary_path = extract_path(completed.stdout, "fake_pipeline_summary_path")
            summary = read_json(summary_path)
            metrics = read_json(summary["metrics_path"])
            table_summary = read_json(summary["table_summary_path"])
            self.assertEqual(summary["execution_env"], "local_wsl")
            self.assertEqual(summary["run_mode"], "smoke_test")
            self.assertEqual(summary["device"], "cpu")
            self.assertFalse(summary["is_paper_result"])
            self.assertTrue(summary["uses_fake_data"])
            self.assertTrue(summary["uses_fake_features"])
            self.assertTrue(summary["fake_or_dry_run"])
            self.assertFalse(metrics["is_paper_result"])
            self.assertTrue(metrics["uses_fake_data"])
            self.assertTrue(metrics["uses_fake_features"])
            self.assertEqual(table_summary["num_eligible_results"], 0)
            for method_name in ["linear_probe", "tip_adapter", "proto_adapter", "rs_cpc"]:
                method_metrics = read_json(summary["method_metrics_paths"][method_name])
                self.assertFalse(method_metrics["is_paper_result"])
                self.assertTrue(method_metrics["fake_or_dry_run"])

            main_accuracy_path = Path(table_summary["outputs"]["main_accuracy"])
            with main_accuracy_path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


if __name__ == "__main__":
    unittest.main()
