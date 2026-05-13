from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export_seed_expansion_plan import export_seed_expansion_plan
from src.utils.io import read_json, safe_write_json


class SeedExpansionPlanTest(unittest.TestCase):
    def test_missing_seed_2_and_3_report_produces_plan_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)

            result = export_plan(root, report_path, target_seeds=[2, 3])

            payload = read_json(result["json_path"])
            self.assertEqual(payload["target_seeds"], [2, 3])
            self.assertEqual(payload["num_plan_items"], 14)
            self.assertEqual(len(payload["plan_items"]), 14)
            self.assertTrue(Path(result["csv_path"]).exists())
            self.assertTrue(Path(result["markdown_path"]).exists())
            seed2_items = [item for item in payload["plan_items"] if item["seed"] == 2]
            self.assertEqual(len(seed2_items), 7)
            self.assertTrue(any("missing_manifest" in item["blocking_reason_from_report"] for item in seed2_items))
            markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
            self.assertIn("This is a planning artifact only.", markdown)
            self.assertIn("It does not run experiments.", markdown)
            self.assertIn("It is not a paper result.", markdown)

    def test_ready_seed_is_not_included_unless_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)

            missing_only = export_plan(root, report_path, target_seeds=[2, 3])
            missing_payload = read_json(missing_only["json_path"])
            self.assertNotIn(1, {item["seed"] for item in missing_payload["plan_items"]})

            ready_requested = export_plan(root, report_path, target_seeds=[1])
            ready_payload = read_json(ready_requested["json_path"])
            self.assertEqual({item["seed"] for item in ready_payload["plan_items"]}, {1})
            by_phase = {item["phase"]: item for item in ready_payload["plan_items"]}
            self.assertEqual(by_phase["B"]["current_status"], "ready")
            self.assertEqual(by_phase["C"]["current_status"], "ready")
            self.assertEqual(by_phase["D"]["current_status"], "ready")
            self.assertEqual(by_phase["E"]["current_status"], "ready")
            self.assertEqual(by_phase["F"]["current_status"], "ready")

    def test_plan_contains_phases_a_through_g(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)

            result = export_plan(root, report_path, target_seeds=[2])

            payload = read_json(result["json_path"])
            self.assertEqual([item["phase"] for item in payload["plan_items"]], ["A", "B", "C", "D", "E", "F", "G"])

    def test_no_result_writing_flags_are_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)

            result = export_plan(root, report_path, target_seeds=[2])

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
            for item in payload["plan_items"]:
                self.assertFalse(item["is_paper_result"])
                self.assertFalse(item["writes_results_raw"])
                self.assertFalse(item["computes_logits"])
                self.assertFalse(item["computes_accuracy"])
                self.assertFalse(item["trains_model"])

    def test_exporter_does_not_modify_input_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)
            before = report_path.read_bytes()

            export_plan(root, report_path, target_seeds=[2, 3])

            self.assertEqual(report_path.read_bytes(), before)

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root)

            with self.assertRaisesRegex(ValueError, "results/raw"):
                export_plan(root, report_path, target_seeds=[2], output_dir=root / "results" / "raw" / "bad")


def export_plan(
    root: Path,
    report_path: Path,
    *,
    target_seeds: list[int],
    output_dir: Path | None = None,
) -> dict[str, object]:
    return export_seed_expansion_plan(
        server_full_preflight_report=report_path,
        dataset="eurosat",
        backbone="remoteclip_vit_b32",
        target_seeds=target_seeds,
        shots=[1, 2, 4, 8, 16],
        output_dir=output_dir or root / "outputs" / "analysis" / "seed_expansion_plans",
        execution_env="remote_server",
        run_mode="local_validation",
        command="pytest seed expansion plan",
    )


def write_fake_server_full_report(root: Path) -> Path:
    seed_status = {
        "1": ready_seed_status(),
        "2": missing_seed_status(2),
        "3": missing_seed_status(3),
    }
    report = {
        "is_valid": False,
        "is_ready_for_server_full": False,
        "errors": ["seed2: missing_manifest", "seed3: missing_manifest"],
        "warnings": [],
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "seeds": [1, 2, 3],
        "shots": [1, 2, 4, 8, 16],
        "methods": ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"],
        "expected_num_runs": 132,
        "ready_num_runs": 44,
        "seed_artifact_summary": seed_status,
        "expected_run_matrix": [],
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
    }
    path = root / "outputs" / "preflight" / "server_full_protocol" / "eurosat_remoteclip_vit_b32" / "report.json"
    return safe_write_json(path, report)


def ready_seed_status() -> dict[str, object]:
    return {
        "manifest_path": "outputs/manifests/feature_cache_after_seed1_support/feature_cache_manifest.json",
        "manifest_exists": True,
        "base_cache_paths": {
            "val": "outputs/features/remoteclip_vit_b32/eurosat/base_seed1/val/feature_cache.pt",
            "test": "outputs/features/remoteclip_vit_b32/eurosat/base_seed1/test/feature_cache.pt",
        },
        "support_cache_paths": {str(shot): f"outputs/features/remoteclip_vit_b32/eurosat/shot_{shot}_seed1/support/feature_cache.pt" for shot in [1, 2, 4, 8, 16]},
        "text_cache_status": {"is_ready": True, "blocking_reason": "", "selected_text_cache_path": "text_feature_cache.pt"},
        "adapter_preflight_status": {"is_ready": True, "blocking_reason": "", "path": "adapter_input_preflight_report.json"},
        "adapter_plan_status": {"is_ready": True, "blocking_reason": "", "path": "adapter_input_plan.json"},
        "prototype_preflight_status": {"is_ready": True, "blocking_reason": "", "path": "rs_cpc_prototype_preflight_report.json"},
        "seed_errors": [],
        "warnings": [],
    }


def missing_seed_status(seed: int) -> dict[str, object]:
    return {
        "manifest_path": None,
        "manifest_exists": False,
        "base_cache_paths": {"val": None, "test": None},
        "support_cache_paths": {str(shot): None for shot in [1, 2, 4, 8, 16]},
        "text_cache_status": {"is_ready": False, "blocking_reason": "missing_text_cache", "selected_text_cache_path": None},
        "adapter_preflight_status": {"is_ready": False, "blocking_reason": "adapter_input_preflight_missing", "path": None},
        "adapter_plan_status": {"is_ready": False, "blocking_reason": "adapter_input_plan_missing", "path": None},
        "prototype_preflight_status": {"is_ready": False, "blocking_reason": "rs_cpc_prototype_preflight_missing", "path": None},
        "seed_errors": [
            "missing_manifest",
            "missing_base_val_cache",
            "missing_base_test_cache",
            "missing_support_cache_shot_1",
            "missing_support_cache_shot_2",
            "missing_support_cache_shot_4",
            "missing_support_cache_shot_8",
            "missing_support_cache_shot_16",
            "missing_text_cache",
            "adapter_input_preflight_missing",
            "adapter_input_plan_missing",
            "rs_cpc_prototype_preflight_missing",
        ],
        "warnings": [],
        "seed": seed,
    }


if __name__ == "__main__":
    unittest.main()
