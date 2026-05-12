from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.export_adapter_input_plan import export_adapter_input_plan
from src.utils.io import read_json, safe_write_json


class ExportAdapterInputPlanTest(unittest.TestCase):
    def test_fake_report_exports_json_and_csv_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_preflight_report(root)

            result = export_adapter_input_plan(
                preflight_report_path=report_path,
                output_dir=root / "outputs" / "preflight" / "adapter_input_plans",
                command="pytest export adapter input plan",
            )

            plan = read_json(result["plan_json_path"])
            rows = read_csv(result["plan_csv_path"])
            self.assertTrue(result["plan_json_path"].exists())
            self.assertTrue(result["plan_csv_path"].exists())
            self.assertFalse(plan["is_paper_result"])
            self.assertFalse(plan["trains_model"])
            self.assertFalse(plan["evaluates_model"])
            self.assertFalse(plan["computes_logits"])
            self.assertFalse(plan["computes_accuracy"])
            self.assertFalse(plan["saves_predictions"])
            self.assertFalse(plan["writes_results_raw"])
            self.assertEqual(plan["source_preflight_report"], str(report_path))
            self.assertEqual(plan["dataset"], "eurosat")
            self.assertEqual(plan["backbone"], "remoteclip_vit_b32")
            self.assertEqual(plan["seed"], "seed1")
            self.assertEqual(plan["num_rows"], 10)
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["dataset"], "eurosat")
            self.assertEqual(rows[0]["feature_dim"], "512")
            self.assertFalse((root / "results" / "raw").exists())

    def test_rs_cpc_m_greater_than_support_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_preflight_report(root)

            result = export_adapter_input_plan(
                preflight_report_path=report_path,
                output_dir=root / "outputs" / "preflight" / "adapter_input_plans",
                command="pytest export adapter input plan m check",
            )

            plan = read_json(result["plan_json_path"])
            rs_cpc_rows = [
                row
                for row in plan["rows"]
                if row["method"] == "rs_cpc" and row["shot_split"] == "shot_1_seed1"
            ]
            by_m = {row["candidate_M"]: row for row in rs_cpc_rows}
            self.assertTrue(by_m[1]["is_ready"])
            self.assertFalse(by_m[2]["is_ready"])
            self.assertEqual(by_m[2]["skip_reason"], "candidate_M_exceeds_min_support_per_class")
            self.assertFalse(by_m[4]["is_ready"])
            self.assertFalse(by_m[8]["is_ready"])

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_preflight_report(root)

            with self.assertRaisesRegex(ValueError, "results/raw"):
                export_adapter_input_plan(
                    preflight_report_path=report_path,
                    output_dir=root / "results" / "raw" / "adapter_input_plans",
                    command="pytest reject results raw",
                )


def write_fake_preflight_report(root: Path) -> Path:
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": ["shot_1_seed1: rs_cpc M=2 exceeds min per-class support count 1"],
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "execution_env": "remote_server",
        "run_mode": "local_validation",
        "is_paper_result": False,
        "manifest_path": "outputs/manifests/fake/feature_cache_manifest.json",
        "checked_base_split": {
            "input": "base_seed1",
            "split_id": "base_seed1",
            "split_path": "splits/eurosat/base_split_seed1.json",
            "dataset": "eurosat",
            "seed": 1,
            "shot": None,
            "num_classes": 10,
        },
        "checked_shot_splits": [
            {
                "input": "shot_1_seed1",
                "split_id": "shot_1_seed1",
                "split_path": "splits/eurosat/shot_1_seed1.json",
                "dataset": "eurosat",
                "seed": 1,
                "shot": 1,
                "num_classes": 10,
            },
            {
                "input": "shot_2_seed1",
                "split_id": "shot_2_seed1",
                "split_path": "splits/eurosat/shot_2_seed1.json",
                "dataset": "eurosat",
                "seed": 1,
                "shot": 2,
                "num_classes": 10,
            },
        ],
        "checked_methods": ["tip_adapter", "proto_adapter", "rs_cpc"],
        "feature_dim": 512,
        "num_classes": 10,
        "per_method_input_summary": {
            "tip_adapter": {
                "per_shot": {
                    "shot_1_seed1": {
                        "method_input_ready": True,
                        "shot": 1,
                        "expected_cache_entries": 10,
                        "actual_support_entries": 10,
                    }
                }
            },
            "proto_adapter": {
                "per_shot": {
                    "shot_1_seed1": {
                        "method_input_ready": True,
                        "shot": 1,
                        "expected_cache_entries": 10,
                        "actual_support_entries": 10,
                    }
                }
            },
            "rs_cpc": {
                "m_values": [1, 2, 4, 8],
                "per_shot": {
                    "shot_1_seed1": {
                        "method_input_ready": False,
                        "method_input_ready_by_M": {"1": True, "2": False, "4": False, "8": False},
                        "shot": 1,
                        "min_support_per_class": 1,
                        "expected_cache_entries_by_M": {"1": 10, "2": 20, "4": 40, "8": 80},
                        "actual_support_entries": 10,
                    },
                    "shot_2_seed1": {
                        "method_input_ready": False,
                        "method_input_ready_by_M": {"1": True, "2": True, "4": False, "8": False},
                        "shot": 2,
                        "min_support_per_class": 2,
                        "expected_cache_entries_by_M": {"1": 10, "2": 20, "4": 40, "8": 80},
                        "actual_support_entries": 20,
                    },
                },
            },
        },
    }
    path = root / "outputs" / "preflight" / "adapter_input" / "report.json"
    return safe_write_json(path, report)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
