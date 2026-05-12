from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_rs_cpc import run_cached_rs_cpc_evaluation
from src.utils.io import read_json, safe_write_json
from tests.test_run_cached_adapters import write_image_cache, write_split, write_text_cache


class RunCachedRsCpcTest(unittest.TestCase):
    def test_fake_cached_rs_cpc_m1_mean_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=1, m_value=1, prototype_init="mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            result = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

            metrics = read_json(result["metrics_path"])
            metadata = read_json(result["metadata_path"])
            self.assertEqual(metrics["method"], "rs_cpc")
            self.assertEqual(metrics["M"], 1)
            self.assertEqual(metrics["prototype_init"], "mean")
            self.assertEqual(metrics["cache_entries"], 2)
            self.assertEqual(metrics["trainable_params"], 0)
            self.assertEqual(metrics["training_time_sec"], 0.0)
            self.assertEqual(metrics["top1_acc_by_split"], {"val": 1.0, "test": 1.0})
            self.assertTrue(metrics["computes_logits"])
            self.assertTrue(metrics["computes_accuracy"])
            self.assertTrue(metrics["evaluates_model"])
            self.assertFalse(metrics["trains_model"])
            self.assertFalse(metrics["extracts_features"])
            self.assertFalse(metrics["loads_model"])
            self.assertFalse(metrics["saves_predictions"])
            self.assertTrue(metrics["writes_results_raw"])
            self.assertFalse(metrics["is_paper_result"])
            self.assertFalse(metrics["eligible_for_paper_tables"])
            self.assertFalse(metadata["is_paper_result"])
            self.assertEqual(metrics["support_cache_path"], str(case["support_cache_path"]))
            self.assertEqual(metrics["image_cache_paths"]["val"], str(case["val_cache_path"]))
            self.assertEqual(metrics["text_feature_cache_path"], str(case["text_cache_path"]))
            self.assertEqual(metrics["adapter_input_plan"], str(case["adapter_input_plan_path"]))
            self.assertEqual(metrics["prototype_preflight_report"], str(case["prototype_preflight_report_path"]))
            self.assertTrue((Path(result["run_dir"]) / "config.yaml").exists())
            self.assertTrue((Path(result["run_dir"]) / "log.txt").exists())
            self.assertIn("/rs_cpc/shot_1/M_1/mean/seed_1/", Path(result["run_dir"]).as_posix())
            self.assertFalse((Path(result["run_dir"]) / "predictions.csv").exists())

    def test_fake_cached_rs_cpc_m2_random_group_mean_writes_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="random_group_mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            result = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

            metrics = read_json(result["metrics_path"])
            self.assertEqual(metrics["cache_entries"], 4)
            self.assertEqual(metrics["compressed_cache_entries"], 4)
            self.assertEqual(metrics["original_cache_entries"], 4)
            self.assertFalse(metrics["is_paper_result"])

    def test_fake_cached_rs_cpc_m2_medoid_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="medoid", plan_ready=True, prototype_ready=True, text_dry_run=True)

            first = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))
            second = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

            first_metrics = read_json(first["metrics_path"])
            second_metrics = read_json(second["metrics_path"])
            self.assertEqual(first_metrics["top1_acc_by_split"], second_metrics["top1_acc_by_split"])
            self.assertEqual(first_metrics["cache_entries"], second_metrics["cache_entries"])
            self.assertNotEqual(first["run_dir"], second["run_dir"])

    def test_m_greater_than_shot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=1, m_value=1, prototype_init="random_group_mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "M must not exceed shot"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True, m_value=2))

    def test_mean_m_greater_than_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "mean prototype_init supports only M=1"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

    def test_kmeans_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="kmeans", plan_ready=True, prototype_ready=True, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "kmeans prototype_init is unsupported"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

    def test_fake_text_cache_is_rejected_for_non_dry_run_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=1, m_value=1, prototype_init="mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "dry-run/fake text cache"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=False))

    def test_plan_row_not_ready_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="medoid", plan_ready=False, prototype_ready=True, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "adapter input plan is not ready"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

    def test_prototype_preflight_not_ready_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=2, m_value=2, prototype_init="medoid", plan_ready=True, prototype_ready=False, text_dry_run=True)

            with self.assertRaisesRegex(ValueError, "prototype preflight report is not ready"):
                run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))

    def test_predictions_are_only_saved_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=1, m_value=1, prototype_init="mean", plan_ready=True, prototype_ready=True, text_dry_run=True)

            no_predictions = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True))
            with_predictions = run_cached_rs_cpc_evaluation(**runner_kwargs(root, case, dry_run=True, save_predictions=True))

            self.assertIsNone(no_predictions["prediction_path"])
            self.assertFalse((Path(no_predictions["run_dir"]) / "predictions.csv").exists())
            self.assertTrue(Path(with_predictions["prediction_path"]).exists())
            self.assertTrue(read_json(with_predictions["metrics_path"])["saves_predictions"])

    def test_outputs_preflight_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_rs_cpc_case(root, shot=1, m_value=1, prototype_init="mean", plan_ready=True, prototype_ready=True, text_dry_run=True)
            kwargs = runner_kwargs(root, case, dry_run=True)
            kwargs["output_dir"] = root / "outputs" / "preflight" / "bad_results"

            with self.assertRaisesRegex(ValueError, "outputs/preflight"):
                run_cached_rs_cpc_evaluation(**kwargs)


def runner_kwargs(
    root: Path,
    case: dict[str, object],
    *,
    dry_run: bool,
    m_value: int | None = None,
    save_predictions: bool = False,
) -> dict[str, object]:
    return {
        "config": {"method": {"name": "rs_cpc"}},
        "config_path": "configs/methods/rs_cpc.yaml",
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "shot": case["shot"],
        "seed": 1,
        "m_value": m_value or case["M"],
        "prototype_init": case["prototype_init"],
        "manifest_path": case["manifest_path"],
        "base_split": str(case["base_split_path"]),
        "shot_split": str(case["shot_split_path"]),
        "text_feature_cache_path": case["text_cache_path"],
        "adapter_input_plan": case["adapter_input_plan_path"],
        "prototype_preflight_report": case["prototype_preflight_report_path"],
        "preflight_report": case["preflight_report_path"],
        "eval_splits": ["val", "test"],
        "output_dir": root / "results" / "raw",
        "device": "cpu",
        "execution_env": "local_wsl",
        "run_mode": "local_validation",
        "dry_run": dry_run,
        "max_samples": None,
        "alpha": 1.0,
        "temperature": 1.0,
        "fusion": "fixed_alpha",
        "save_predictions": save_predictions,
        "allow_paper_result": False,
        "command": "pytest cached rs_cpc",
    }


def write_rs_cpc_case(
    root: Path,
    *,
    shot: int,
    m_value: int,
    prototype_init: str,
    plan_ready: bool,
    prototype_ready: bool,
    text_dry_run: bool,
) -> dict[str, object]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 2
    base_split_path = write_split(root, dataset=dataset, split_id="base_seed1", shot=None, num_classes=num_classes)
    shot_split_path = write_split(root, dataset=dataset, split_id=f"shot_{shot}_seed1", shot=shot, num_classes=num_classes)
    entries = [
        write_image_cache(root, dataset=dataset, backbone=backbone, split_id="base_seed1", split_path=base_split_path, section="val", labels=[0, 1]),
        write_image_cache(root, dataset=dataset, backbone=backbone, split_id="base_seed1", split_path=base_split_path, section="test", labels=[0, 1]),
        write_image_cache(
            root,
            dataset=dataset,
            backbone=backbone,
            split_id=f"shot_{shot}_seed1",
            split_path=shot_split_path,
            section="support",
            labels=[label for label in range(num_classes) for _ in range(shot)],
        ),
    ]
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    text_cache_path = root / "features" / backbone / dataset / "base_seed1" / "text" / "run" / "text_feature_cache.pt"
    write_text_cache(text_cache_path, dataset=dataset, backbone=backbone, dry_run=text_dry_run)
    preflight_report_path = write_rs_cpc_adapter_preflight(
        root,
        dataset=dataset,
        backbone=backbone,
        base_split_path=base_split_path,
        shot_split_path=shot_split_path,
        shot=shot,
        m_value=m_value,
        ready=True,
    )
    adapter_input_plan_path = write_rs_cpc_adapter_plan(
        root,
        dataset=dataset,
        backbone=backbone,
        shot_split_path=shot_split_path,
        shot=shot,
        m_value=m_value,
        plan_ready=plan_ready,
    )
    prototype_preflight_report_path = write_rs_cpc_prototype_report(
        root,
        dataset=dataset,
        backbone=backbone,
        shot_split_path=shot_split_path,
        shot=shot,
        m_value=m_value,
        prototype_init=prototype_init,
        prototype_ready=prototype_ready,
    )
    return {
        "shot": shot,
        "M": m_value,
        "prototype_init": prototype_init,
        "manifest_path": manifest_path,
        "base_split_path": base_split_path,
        "shot_split_path": shot_split_path,
        "text_cache_path": text_cache_path,
        "preflight_report_path": preflight_report_path,
        "adapter_input_plan_path": adapter_input_plan_path,
        "prototype_preflight_report_path": prototype_preflight_report_path,
        "support_cache_path": Path(entries[2]["feature_cache_path"]),
        "val_cache_path": Path(entries[0]["feature_cache_path"]),
    }


def write_rs_cpc_adapter_preflight(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    base_split_path: Path,
    shot_split_path: Path,
    shot: int,
    m_value: int,
    ready: bool,
) -> Path:
    path = root / "outputs" / "preflight" / "adapter_input" / "adapter_input_preflight_report.json"
    safe_write_json(
        path,
        {
            "is_valid": ready,
            "dataset": dataset,
            "backbone": backbone,
            "per_split_summary": {
                str(base_split_path): {
                    "split_kind": "base",
                    "sections": {"val": {"is_ready": True}, "test": {"is_ready": True}},
                    "val_ready_for_tuning_input": True,
                    "test_ready_for_evaluation_input": True,
                },
                str(shot_split_path): {
                    "split_kind": "shot",
                    "shot": shot,
                    "support": {"is_ready": True, "num_samples": 2 * shot},
                    "support_balanced": True,
                    "min_support_per_class": shot,
                },
            },
            "per_method_input_summary": {
                "rs_cpc": {
                    "per_shot": {
                        str(shot_split_path): {
                            "method_input_ready_by_M": {str(m_value): ready},
                            "shot": shot,
                            "expected_cache_entries_by_M": {str(m_value): 2 * m_value},
                            "actual_support_entries": 2 * shot,
                        }
                    }
                }
            },
        },
    )
    return path


def write_rs_cpc_adapter_plan(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    shot_split_path: Path,
    shot: int,
    m_value: int,
    plan_ready: bool,
) -> Path:
    path = root / "outputs" / "preflight" / "adapter_input_plans" / "adapter_input_plan.json"
    safe_write_json(
        path,
        {
            "source_preflight_is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "seed": "seed1",
            "num_classes": 2,
            "feature_dim": 2,
            "rows": [
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": "seed1",
                    "shot_split": str(shot_split_path),
                    "shot": shot,
                    "method": "rs_cpc",
                    "support_entries": 2 * shot,
                    "candidate_M": m_value,
                    "is_ready": plan_ready,
                    "skip_reason": "" if plan_ready else "candidate_M_exceeds_min_support_per_class",
                    "expected_cache_entries": 2 * m_value,
                }
            ],
        },
    )
    return path


def write_rs_cpc_prototype_report(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    shot_split_path: Path,
    shot: int,
    m_value: int,
    prototype_init: str,
    prototype_ready: bool,
) -> Path:
    path = root / "outputs" / "preflight" / "rs_cpc_prototypes" / "rs_cpc_prototype_preflight_report.json"
    safe_write_json(
        path,
        {
            "is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "per_combination_summary": [
                {
                    "shot_split": str(shot_split_path),
                    "shot": shot,
                    "candidate_M": m_value,
                    "prototype_init": prototype_init,
                    "prototype_shape": [2 * m_value, 2],
                    "prototype_label_shape": [2 * m_value],
                    "prototype_counts_by_label": {"0": m_value, "1": m_value},
                    "is_ready": prototype_ready,
                }
            ],
        },
    )
    return path


if __name__ == "__main__":
    unittest.main()
