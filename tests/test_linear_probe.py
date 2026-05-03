from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.baselines.linear_probe import LinearProbe
from src.features.feature_cache import make_fake_feature_cache
from src.baselines.runner_utils import split_support_query
from src.utils.io import read_json


class LinearProbeTest(unittest.TestCase):
    def test_linear_probe_synthetic(self) -> None:
        cache = make_fake_feature_cache(num_samples=12, num_classes=3, feature_dim=8)
        split = split_support_query(cache, shot=1)
        method = LinearProbe().fit(split["support_features"], split["support_labels"])
        logits = method.predict_logits(split["query_features"])
        self.assertEqual(len(logits), len(split["query_labels"]))
        self.assertEqual(len(logits[0]), 3)

    def test_linear_probe_script_dry_run_non_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [sys.executable, "scripts/run_linear_probe.py", "--dataset", "eurosat", "--backbone", "fake_backbone", "--dry-run", "--max-samples", "12", "--execution-env", "local_wsl", "--run-mode", "smoke_test", "--device", "cpu", "--output-dir", temp_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = read_json(extract_path(completed.stdout, "metrics_path"))
            self.assertFalse(metrics["is_paper_result"])
            self.assertTrue(metrics["fake_or_dry_run"])


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(stdout)


if __name__ == "__main__":
    unittest.main()
