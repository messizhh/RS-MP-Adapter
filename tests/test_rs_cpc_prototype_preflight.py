from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_rs_cpc_prototype_preflight import run_rs_cpc_prototype_preflight
from src.features.feature_cache import FeatureCache, save_feature_cache
from src.utils.io import read_json, safe_write_json


class RSCPCPrototypePreflightTest(unittest.TestCase):
    def test_ready_rows_generate_prototype_shape_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_fake_inputs(root, include_m_gt_shot_ready=False)

            report_path, is_valid = run_rs_cpc_prototype_preflight(
                adapter_input_plan_path=inputs["plan_path"],
                preflight_report_path=inputs["preflight_report_path"],
                prototype_inits=["random_group_mean", "medoid"],
                output_dir=root / "outputs" / "preflight" / "rs_cpc_prototypes",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest rs cpc prototype preflight",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["errors"], [])
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["writes_results_raw"])
            self.assertEqual(len(report["checked_rows"]), 4)
            summaries = {
                (row["prototype_init"], row["candidate_M"]): row for row in report["per_combination_summary"]
            }
            self.assertEqual(summaries[("random_group_mean", 2)]["prototype_shape"], [6, 4])
            self.assertEqual(summaries[("random_group_mean", 2)]["prototype_label_shape"], [6])
            self.assertEqual(
                summaries[("random_group_mean", 2)]["prototype_counts_by_label"],
                {"0": 2, "1": 2, "2": 2},
            )
            self.assertTrue(summaries[("medoid", 2)]["prototypes_finite"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_mean_m_greater_than_one_is_skipped_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_fake_inputs(root, include_m_gt_shot_ready=False)

            report_path, is_valid = run_rs_cpc_prototype_preflight(
                adapter_input_plan_path=inputs["plan_path"],
                preflight_report_path=inputs["preflight_report_path"],
                prototype_inits=["mean"],
                output_dir=root / "outputs" / "preflight" / "rs_cpc_prototypes",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest rs cpc prototype mean skip",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertEqual(len(report["checked_rows"]), 1)
            self.assertTrue(
                any(row["skip_reason"] == "mean_unsupported_for_M_gt_1" for row in report["skipped_rows"])
            )
            self.assertTrue(any("mean init supports only M=1" in warning for warning in report["warnings"]))

    def test_m_greater_than_shot_ready_plan_row_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_fake_inputs(root, include_m_gt_shot_ready=True)

            report_path, is_valid = run_rs_cpc_prototype_preflight(
                adapter_input_plan_path=inputs["plan_path"],
                preflight_report_path=inputs["preflight_report_path"],
                prototype_inits=["random_group_mean"],
                output_dir=root / "outputs" / "preflight" / "rs_cpc_prototypes",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest rs cpc prototype m greater than shot",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertFalse(any(row["candidate_M"] == 4 for row in report["checked_rows"]))
            self.assertTrue(
                any(row["skip_reason"] == "candidate_M_exceeds_shot_from_plan" for row in report["skipped_rows"])
            )
            self.assertFalse((root / "results" / "raw").exists())

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_fake_inputs(root, include_m_gt_shot_ready=False)

            with self.assertRaisesRegex(ValueError, "results/raw"):
                run_rs_cpc_prototype_preflight(
                    adapter_input_plan_path=inputs["plan_path"],
                    preflight_report_path=inputs["preflight_report_path"],
                    prototype_inits=["random_group_mean"],
                    output_dir=root / "results" / "raw" / "rs_cpc_prototypes",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    command="pytest reject prototype results raw",
                )


def write_fake_inputs(root: Path, *, include_m_gt_shot_ready: bool) -> dict[str, Path]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 3
    feature_dim = 4
    shot = 2
    cache_path = write_support_cache(
        root / "features" / backbone / dataset / "shot_2_seed1" / "support" / "feature_cache.pt",
        dataset=dataset,
        backbone=backbone,
        num_classes=num_classes,
        feature_dim=feature_dim,
        shot=shot,
    )
    preflight_report_path = root / "outputs" / "preflight" / "adapter_input" / "report.json"
    safe_write_json(
        preflight_report_path,
        {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "dataset": dataset,
            "backbone": backbone,
            "execution_env": "local_wsl",
            "run_mode": "local_validation",
            "is_paper_result": False,
            "manifest_path": "outputs/manifests/fake/feature_cache_manifest.json",
            "checked_base_split": {"input": "base_seed1", "seed": 1},
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "per_split_summary": {
                "shot_2_seed1": {
                    "split_kind": "shot",
                    "split_id": "shot_2_seed1",
                    "shot": shot,
                    "support": {
                        "cache_path": str(cache_path),
                        "num_samples": num_classes * shot,
                        "feature_dim": feature_dim,
                        "num_classes": num_classes,
                    },
                    "support_balanced": True,
                    "min_support_per_class": shot,
                }
            },
        },
    )
    rows = [
        plan_row("shot_2_seed1", shot, 1, True, num_classes),
        plan_row("shot_2_seed1", shot, 2, True, num_classes),
        plan_row("shot_2_seed1", shot, 4, include_m_gt_shot_ready, num_classes),
    ]
    plan_path = root / "outputs" / "preflight" / "adapter_input_plans" / "plan.json"
    safe_write_json(
        plan_path,
        {
            "is_paper_result": False,
            "source_preflight_report": str(preflight_report_path),
            "dataset": dataset,
            "backbone": backbone,
            "seed": "seed1",
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "rows": rows,
        },
    )
    return {"plan_path": plan_path, "preflight_report_path": preflight_report_path, "cache_path": cache_path}


def plan_row(shot_split: str, shot: int, candidate_m: int, is_ready: bool, num_classes: int) -> dict[str, object]:
    return {
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "seed": "seed1",
        "shot_split": shot_split,
        "shot": shot,
        "method": "rs_cpc",
        "num_classes": num_classes,
        "feature_dim": 4,
        "support_entries": num_classes * shot,
        "candidate_M": candidate_m,
        "is_ready": is_ready,
        "skip_reason": "" if is_ready else "candidate_M_exceeds_min_support_per_class",
        "expected_cache_entries": num_classes * candidate_m,
    }


def write_support_cache(
    path: Path,
    *,
    dataset: str,
    backbone: str,
    num_classes: int,
    feature_dim: int,
    shot: int,
) -> Path:
    labels = [label for label in range(num_classes) for _ in range(shot)]
    features = []
    for index, label in enumerate(labels):
        features.append([float(label + 1), float(index + 1), 0.5, 1.0][:feature_dim])
    cache = FeatureCache(
        image_features=features,
        image_labels=labels,
        image_paths=[f"fake://support/{idx}.jpg" for idx in range(len(labels))],
        split_name="support",
        class_to_idx={f"class_{idx}": idx for idx in range(num_classes)},
        backbone=backbone,
        dataset=dataset,
        feature_dim=feature_dim,
        normalize_features=True,
        created_at="2026-05-12T00:00:00+00:00",
        source_script="tests/test_rs_cpc_prototype_preflight.py",
        metadata={"is_paper_result": False},
    )
    return save_feature_cache(cache, path)


if __name__ == "__main__":
    unittest.main()
