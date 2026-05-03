from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest

from src.adapters.rs_cpc_adapter import RsCpcAdapter
from src.baselines.runner_utils import split_support_query
from src.features.feature_cache import make_fake_feature_cache
from src.utils.io import read_json


class RsCpcTest(unittest.TestCase):
    def test_rs_cpc_cache_entries(self) -> None:
        cache = make_fake_feature_cache(num_samples=12, num_classes=3, feature_dim=8)
        split = split_support_query(cache, shot=4)
        method = RsCpcAdapter(num_prototypes_per_class=2, prototype_init="random_group_mean", seed=1)
        method.fit(split["support_features"], split["support_labels"])
        self.assertEqual(method.cache_entries, 6)
        self.assertEqual(method.compression_ratio, 2.0)

    def test_rs_cpc_script_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [sys.executable, "scripts/run_rs_cpc.py", "--dataset", "eurosat", "--backbone", "fake_backbone", "--dry-run", "--max-samples", "12", "--num-prototypes-per-class", "2", "--prototype-init", "random_group_mean", "--execution-env", "local_wsl", "--run-mode", "smoke_test", "--device", "cpu", "--output-dir", temp_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            metrics_path = [line.split("=", 1)[1] for line in completed.stdout.splitlines() if line.startswith("metrics_path=")][0]
            metrics = read_json(metrics_path)
            self.assertEqual(metrics["cache_entries"], 6)
            self.assertFalse(metrics["is_paper_result"])


if __name__ == "__main__":
    unittest.main()
