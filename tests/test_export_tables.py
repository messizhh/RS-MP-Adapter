from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_json, safe_write_json


class ExportTablesTest(unittest.TestCase):
    def test_export_tables_excludes_smoke_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "raw" / "smoke" / "metrics.json"
            safe_write_json(
                metrics_path,
                {
                    "dataset": "eurosat",
                    "shot": 1,
                    "backbone": "fake_backbone",
                    "method": "zero_shot_clip",
                    "seed": 1,
                    "top1_acc": 0.5,
                    "run_mode": "smoke_test",
                    "execution_env": "local_wsl",
                    "is_paper_result": False,
                    "fake_or_dry_run": True,
                    "uses_fake_data": True,
                    "uses_fake_features": True,
                },
            )
            output_dir = root / "tables"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_tables.py",
                    "--input-dir",
                    str(root / "raw"),
                    "--output-dir",
                    str(output_dir),
                    "--tables",
                    "main",
                    "efficiency",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary_path = extract_path(completed.stdout, "summary_path")
            summary = read_json(summary_path)
            self.assertEqual(summary["num_result_json"], 1)
            self.assertEqual(summary["num_eligible_results"], 0)
            with (output_dir / "main_accuracy.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_export_tables_includes_server_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "raw" / "server" / "metrics.json"
            safe_write_json(
                metrics_path,
                {
                    "dataset": "eurosat",
                    "shot": 1,
                    "backbone": "remoteclip_vit_b32",
                    "method": "zero_shot_clip",
                    "seed": 1,
                    "top1_acc": 0.5,
                    "cache_entries": 0,
                    "trainable_params": 0,
                    "training_time_sec": 0.0,
                    "inference_time_sec": 1.0,
                    "images_per_second": 10.0,
                    "gpu_memory_mb": None,
                    "run_mode": "server_full",
                    "execution_env": "remote_server",
                    "is_paper_result": True,
                    "fake_or_dry_run": False,
                    "uses_fake_data": False,
                    "uses_fake_features": False,
                },
            )
            output_dir = root / "tables"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/export_tables.py",
                    "--input-dir",
                    str(root / "raw"),
                    "--output-dir",
                    str(output_dir),
                    "--tables",
                    "main",
                    "efficiency",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output_dir / "main_accuracy.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_mode"], "server_full")


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


if __name__ == "__main__":
    unittest.main()
