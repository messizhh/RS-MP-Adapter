from __future__ import annotations

import re
import unittest
from pathlib import Path


SERVER_SCRIPTS = [
    Path("scripts/server/check_server_preflight.sh"),
    Path("scripts/server/run_phase1_baselines.sh"),
    Path("scripts/server/run_rs_cpc_sweep.sh"),
    Path("scripts/server/run_ablation.sh"),
    Path("scripts/server/export_tables.sh"),
]

SERVER_RUNNER_TEMPLATES = [
    Path("scripts/server/run_phase1_baselines.sh"),
    Path("scripts/server/run_rs_cpc_sweep.sh"),
    Path("scripts/server/run_ablation.sh"),
]


class ServerScriptTemplateTest(unittest.TestCase):
    def test_server_scripts_are_templates_with_todo_placeholders(self) -> None:
        for script in SERVER_SCRIPTS:
            with self.subTest(script=str(script)):
                text = script.read_text(encoding="utf-8")
                self.assertIn("Template only", text)
                self.assertIn("manually", text)
                for placeholder in ["TODO_DATASET_ROOT", "TODO_FEATURE_ROOT", "TODO_CHECKPOINT_ROOT", "TODO_RESULT_ROOT", "TODO_LOG_ROOT"]:
                    self.assertIn(placeholder, text)

    def test_server_runner_scripts_use_remote_cuda_flags(self) -> None:
        for script in SERVER_RUNNER_TEMPLATES:
            with self.subTest(script=str(script)):
                text = script.read_text(encoding="utf-8")
                self.assertIn("--execution-env remote_server", text)
                self.assertIn("--device cuda", text)
                self.assertRegex(text, r"--run-mode server_(full|ablation|benchmark)")

    def test_server_preflight_is_non_experimental(self) -> None:
        text = Path("scripts/server/check_server_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("Non-experimental server preflight", text)
        self.assertIn("torch_available", text)
        self.assertIn("cuda_available", text)
        self.assertIn("RUN_DATASET_LAYOUT_CHECK", text)
        self.assertIn("scripts/check_dataset_layout.py", text)
        self.assertIn("WEIGHT_ROOT is deprecated", text)
        self.assertIn("CHECKPOINT_ROOT", text)
        self.assertIn("RESULT_ROOT", text)
        self.assertIn("LOG_ROOT", text)

    def test_server_export_template_keeps_server_filters(self) -> None:
        text = Path("scripts/server/export_tables.sh").read_text(encoding="utf-8")
        self.assertIn("server_full server_ablation server_benchmark", text)
        self.assertIn("dry_run smoke_test debug tiny_subset local_validation", text)

    def test_server_scripts_do_not_contain_private_absolute_paths(self) -> None:
        private_path_pattern = re.compile(r"(/home/|/mnt/|/Users/|[A-Za-z]:\\\\)")
        for script in SERVER_SCRIPTS:
            with self.subTest(script=str(script)):
                text = "\n".join(line for line in script.read_text(encoding="utf-8").splitlines() if not line.startswith("#!"))
                self.assertIsNone(private_path_pattern.search(text))


if __name__ == "__main__":
    unittest.main()
