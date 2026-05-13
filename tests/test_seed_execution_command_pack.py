from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export_seed_execution_command_pack import export_seed_execution_command_pack
from src.utils.io import read_json, safe_write_json


class SeedExecutionCommandPackTest(unittest.TestCase):
    def test_fake_seed_expansion_plan_generates_seed2_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)

            result = export_pack(root, plan_path)

            self.assertTrue(Path(result["markdown_path"]).exists())
            self.assertTrue(Path(result["shell_path"]).exists())
            self.assertTrue(Path(result["json_path"]).exists())
            payload = read_json(result["json_path"])
            self.assertEqual(payload["dataset"], "eurosat")
            self.assertEqual(payload["backbone"], "remoteclip_vit_b32")
            self.assertEqual(payload["seed"], 2)
            self.assertEqual(payload["num_command_items"], 7)
            self.assertTrue(payload["script_inventory"]["generate_splits"]["exists"])
            markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
            self.assertIn("This command pack is generated for review only.", markdown)
            self.assertIn("It does not run experiments.", markdown)
            self.assertIn("It is not a paper result.", markdown)

    def test_shell_pack_contains_exit_1_before_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)

            result = export_pack(root, plan_path)

            shell_text = Path(result["shell_path"]).read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", shell_text)
            self.assertIn('echo "This command pack is generated for review. Do not run blindly."', shell_text)
            self.assertLess(shell_text.index("exit 1"), shell_text.index("python3 scripts/generate_splits.py"))

    def test_command_pack_uses_seed2_paths_not_seed1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)

            result = export_pack(root, plan_path)

            combined = "\n".join(
                [
                    Path(result["markdown_path"]).read_text(encoding="utf-8"),
                    Path(result["shell_path"]).read_text(encoding="utf-8"),
                    Path(result["json_path"]).read_text(encoding="utf-8"),
                ]
            )
            self.assertIn("base_seed2", combined)
            self.assertIn("shot_16_seed2", combined)
            self.assertIn("feature_cache_after_seed2_support", combined)
            self.assertIn("adapter_input/eurosat_remoteclip_vit_b32_seed2", combined)
            self.assertIn("adapter_input_plans/eurosat_remoteclip_vit_b32_seed2", combined)
            self.assertIn("rs_cpc_prototypes/eurosat_remoteclip_vit_b32_seed2", combined)
            self.assertNotIn("base_seed1", combined)
            self.assertNotIn("shot_1_seed1", combined)
            self.assertNotIn("feature_cache_after_seed1_support", combined)

    def test_command_pack_includes_phases_a_through_g(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)

            result = export_pack(root, plan_path)

            payload = read_json(result["json_path"])
            self.assertEqual([item["phase"] for item in payload["command_items"]], ["A", "B", "C", "D", "E", "F", "G"])

    def test_no_result_writing_flags_are_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)

            result = export_pack(root, plan_path)

            payload = read_json(result["json_path"])
            for key in [
                "is_paper_result",
                "writes_results_raw",
                "computes_logits",
                "computes_accuracy",
                "evaluates_model",
                "trains_model",
                "modifies_results",
                "deletes_results",
            ]:
                self.assertFalse(payload[key])
            for item in payload["command_items"]:
                self.assertFalse(item["is_paper_result"])
                self.assertFalse(item["writes_results_raw"])
                self.assertFalse(item["computes_logits"])
                self.assertFalse(item["computes_accuracy"])
                self.assertFalse(item["evaluates_model"])
                self.assertFalse(item["trains_model"])

    def test_input_plan_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = write_fake_seed_expansion_plan(root)
            before = plan_path.read_bytes()

            export_pack(root, plan_path)

            self.assertEqual(plan_path.read_bytes(), before)


def export_pack(root: Path, plan_path: Path) -> dict[str, object]:
    return export_seed_execution_command_pack(
        seed_expansion_plan=plan_path,
        dataset="eurosat",
        backbone="remoteclip_vit_b32",
        seed=2,
        shots=[1, 2, 4, 8, 16],
        output_dir=root / "outputs" / "analysis" / "seed_execution_command_packs",
        execution_env="remote_server",
        run_mode="local_validation",
        command="pytest seed execution command pack",
    )


def write_fake_seed_expansion_plan(root: Path) -> Path:
    plan = {
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "target_seeds": [2, 3],
        "shots": [1, 2, 4, 8, 16],
        "num_plan_items": 14,
        "plan_items": fake_plan_items(seed=2) + fake_plan_items(seed=3),
    }
    path = root / "outputs" / "analysis" / "seed_expansion_plans" / "eurosat_remoteclip_vit_b32_seed2_seed3" / "plan" / "seed_expansion_plan.json"
    return safe_write_json(path, plan)


def fake_plan_items(*, seed: int) -> list[dict[str, object]]:
    phases = [
        ("A", "dataset/split readiness"),
        ("B", "image feature cache manifest and support caches"),
        ("C", "standalone text feature cache"),
        ("D", "adapter input preflight"),
        ("E", "adapter input plan"),
        ("F", "RS-CPC prototype preflight"),
        ("G", "rerun server_full protocol preflight"),
    ]
    return [
        {
            "seed": seed,
            "phase": phase,
            "artifact_type": artifact,
            "expected_path_or_pattern": f"seed{seed}:{artifact}",
            "current_status": "missing",
            "blocking_reason_from_report": [f"missing_{phase}"],
            "suggested_script": "placeholder",
            "suggested_command_template": "placeholder",
            "is_paper_result": False,
            "writes_results_raw": False,
            "computes_logits": False,
            "computes_accuracy": False,
            "trains_model": False,
        }
        for phase, artifact in phases
    ]


if __name__ == "__main__":
    unittest.main()
