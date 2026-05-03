from __future__ import annotations

import subprocess
import sys
import unittest


COMMON_OPTIONS = {
    "--config",
    "--dataset",
    "--backbone",
    "--method",
    "--shot",
    "--seed",
    "--split",
    "--feature-cache",
    "--output-dir",
    "--dry-run",
    "--max-samples",
    "--device",
    "--execution-env",
    "--run-mode",
}


class RunnerCliTest(unittest.TestCase):
    def test_major_runner_scripts_expose_common_options(self) -> None:
        scripts = [
            "scripts/run_zero_shot.py",
            "scripts/run_linear_probe.py",
            "scripts/run_tip_adapter.py",
            "scripts/run_proto_adapter.py",
            "scripts/run_rs_cpc.py",
            "scripts/extract_features.py",
            "scripts/generate_splits.py",
            "scripts/inspect_dataset.py",
            "scripts/run_fake_pipeline.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                help_text = run_help(script)
                missing = sorted(option for option in COMMON_OPTIONS if option not in help_text)
                self.assertEqual(missing, [])

    def test_finetune_flags_are_exposed_but_disabled(self) -> None:
        for script in ["scripts/run_linear_probe.py", "scripts/run_tip_adapter.py", "scripts/run_proto_adapter.py", "scripts/run_rs_cpc.py"]:
            with self.subTest(script=script):
                help_text = run_help(script)
                self.assertIn("--finetune", help_text)
                self.assertIn("--resume", help_text)
                self.assertIn("--checkpoint", help_text)

    def test_export_tables_exposes_filter_options(self) -> None:
        help_text = run_help("scripts/export_tables.py")
        for option in [
            "--input-dir",
            "--output-dir",
            "--tables",
            "--include-run-modes",
            "--exclude-run-modes",
            "--allow-local-results",
            "--include-fake-results",
        ]:
            self.assertIn(option, help_text)


def run_help(script: str) -> str:
    completed = subprocess.run([sys.executable, script, "--help"], check=True, capture_output=True, text=True)
    return completed.stdout


if __name__ == "__main__":
    unittest.main()
