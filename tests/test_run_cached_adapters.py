from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.run_proto_adapter import run_cached_proto_adapter_evaluation
from scripts.run_tip_adapter import run_cached_tip_adapter_evaluation
from src.utils.io import read_json, safe_write_json


class RunCachedAdaptersTest(unittest.TestCase):
    def test_fake_cached_tip_adapter_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=2)

            result = run_cached_tip_adapter_evaluation(**runner_kwargs(root, case, method="tip_adapter", dry_run=True))

            metrics = read_json(result["metrics_path"])
            metadata = read_json(result["metadata_path"])
            self.assertEqual(metrics["method"], "tip_adapter")
            self.assertEqual(metrics["cache_entries"], 4)
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
            self.assertIn("git_commit", metrics)
            self.assertIn("python_version", metrics)
            self.assertIn("torch_version", metrics)
            self.assertIn("cuda_version", metrics)
            self.assertIn("gpu_name", metrics)
            self.assertEqual(metrics["command"], "pytest cached tip_adapter")
            self.assertEqual(metrics["support_cache_path"], str(case["support_cache_path"]))
            self.assertEqual(metrics["image_cache_paths"]["val"], str(case["val_cache_path"]))
            self.assertEqual(metrics["text_feature_cache_path"], str(case["text_cache_path"]))
            self.assertTrue((Path(result["run_dir"]) / "config.yaml").exists())
            self.assertTrue((Path(result["run_dir"]) / "log.txt").exists())
            self.assertFalse((Path(result["run_dir"]) / "predictions.csv").exists())

    def test_fake_cached_proto_adapter_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=2)

            result = run_cached_proto_adapter_evaluation(**runner_kwargs(root, case, method="proto_adapter", dry_run=True))

            metrics = read_json(result["metrics_path"])
            self.assertEqual(metrics["method"], "proto_adapter")
            self.assertEqual(metrics["cache_entries"], 2)
            self.assertEqual(metrics["trainable_params"], 0)
            self.assertEqual(metrics["top1_acc_by_split"], {"val": 1.0, "test": 1.0})
            self.assertFalse(metrics["is_paper_result"])

    def test_fake_text_cache_is_rejected_for_non_dry_run_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=1)

            with self.assertRaisesRegex(ValueError, "dry-run/fake text cache"):
                run_cached_tip_adapter_evaluation(**runner_kwargs(root, case, method="tip_adapter", dry_run=False))

    def test_adapter_input_plan_not_ready_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=False, text_dry_run=True, shot=1)

            with self.assertRaisesRegex(ValueError, "adapter input plan is not ready"):
                run_cached_tip_adapter_evaluation(**runner_kwargs(root, case, method="tip_adapter", dry_run=True))

    def test_adapter_input_preflight_not_ready_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=False, plan_ready=True, text_dry_run=True, shot=1)

            with self.assertRaisesRegex(ValueError, "adapter input preflight report is not ready"):
                run_cached_proto_adapter_evaluation(**runner_kwargs(root, case, method="proto_adapter", dry_run=True))

    def test_outputs_preflight_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=1)
            kwargs = runner_kwargs(root, case, method="tip_adapter", dry_run=True)
            kwargs["output_dir"] = root / "outputs" / "preflight" / "bad_results"

            with self.assertRaisesRegex(ValueError, "outputs/preflight"):
                run_cached_tip_adapter_evaluation(**kwargs)

    def test_predictions_are_only_saved_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=1)

            no_predictions = run_cached_tip_adapter_evaluation(**runner_kwargs(root, case, method="tip_adapter", dry_run=True))
            with_predictions = run_cached_tip_adapter_evaluation(
                **runner_kwargs(root, case, method="tip_adapter", dry_run=True, save_predictions=True)
            )

            self.assertIsNone(no_predictions["prediction_path"])
            self.assertFalse((Path(no_predictions["run_dir"]) / "predictions.csv").exists())
            self.assertTrue(Path(with_predictions["prediction_path"]).exists())
            self.assertTrue(read_json(with_predictions["metrics_path"])["saves_predictions"])

    def test_run_directory_is_unique_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_adapter_case(root, method_ready=True, plan_ready=True, text_dry_run=True, shot=1)
            kwargs = runner_kwargs(root, case, method="proto_adapter", dry_run=True)

            first = run_cached_proto_adapter_evaluation(**kwargs)
            second = run_cached_proto_adapter_evaluation(**kwargs)

            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertTrue(Path(first["metrics_path"]).exists())
            self.assertTrue(Path(second["metrics_path"]).exists())


def runner_kwargs(
    root: Path,
    case: dict[str, Path],
    *,
    method: str,
    dry_run: bool,
    save_predictions: bool = False,
) -> dict[str, object]:
    return {
        "config": {"method": {"name": method}},
        "config_path": f"configs/methods/{method}.yaml",
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "shot": case["shot"],
        "seed": 1,
        "manifest_path": case["manifest_path"],
        "base_split": str(case["base_split_path"]),
        "shot_split": str(case["shot_split_path"]),
        "text_feature_cache_path": case["text_cache_path"],
        "adapter_input_plan": case["adapter_input_plan_path"],
        "eval_splits": ["val", "test"],
        "output_dir": root / "results" / "raw",
        "device": "cpu",
        "execution_env": "local_wsl",
        "run_mode": "local_validation",
        "preflight_report": case["preflight_report_path"],
        "dry_run": dry_run,
        "max_samples": None,
        "alpha": 1.0,
        "temperature": 1.0,
        "save_predictions": save_predictions,
        "allow_paper_result": False,
        "command": f"pytest cached {method}",
    }


def write_cached_adapter_case(
    root: Path,
    *,
    method_ready: bool,
    plan_ready: bool,
    text_dry_run: bool,
    shot: int,
) -> dict[str, Path]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 2
    base_split_path = write_split(root, dataset=dataset, split_id="base_seed1", shot=None, num_classes=num_classes)
    shot_split_path = write_split(root, dataset=dataset, split_id=f"shot_{shot}_seed1", shot=shot, num_classes=num_classes)
    entries = [
        write_image_cache(
            root,
            dataset=dataset,
            backbone=backbone,
            split_id="base_seed1",
            split_path=base_split_path,
            section="val",
            labels=[0, 1],
        ),
        write_image_cache(
            root,
            dataset=dataset,
            backbone=backbone,
            split_id="base_seed1",
            split_path=base_split_path,
            section="test",
            labels=[0, 1],
        ),
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
    preflight_report_path = write_adapter_preflight_report(
        root,
        dataset=dataset,
        backbone=backbone,
        base_split_path=base_split_path,
        shot_split_path=shot_split_path,
        shot=shot,
        method_ready=method_ready,
    )
    adapter_input_plan_path = write_adapter_input_plan(
        root,
        dataset=dataset,
        backbone=backbone,
        shot_split_path=shot_split_path,
        shot=shot,
        plan_ready=plan_ready,
    )
    return {
        "shot": shot,
        "manifest_path": manifest_path,
        "base_split_path": base_split_path,
        "shot_split_path": shot_split_path,
        "text_cache_path": text_cache_path,
        "preflight_report_path": preflight_report_path,
        "adapter_input_plan_path": adapter_input_plan_path,
        "support_cache_path": Path(entries[2]["feature_cache_path"]),
        "val_cache_path": Path(entries[0]["feature_cache_path"]),
    }


def write_split(root: Path, *, dataset: str, split_id: str, shot: int | None, num_classes: int) -> Path:
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    rows = [{"class_name": f"class_{idx}", "label": idx, "path": f"class_{idx}/{idx}.jpg"} for idx in range(num_classes)]
    support = [
        {"class_name": f"class_{label}", "label": label, "path": f"class_{label}/support_{idx}.jpg"}
        for label in range(num_classes)
        for idx in range(shot or 0)
    ]
    path = root / "splits" / dataset / f"{split_id}.json"
    safe_write_json(
        path,
        {
            "dataset": dataset,
            "seed": 1,
            "shot": shot,
            "class_to_idx": class_to_idx,
            "num_classes": num_classes,
            "train": rows,
            "val": rows,
            "test": rows,
            "support": support,
        },
    )
    return path


def write_image_cache(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    split_id: str,
    split_path: Path,
    section: str,
    labels: list[int],
) -> dict[str, str]:
    run_dir = root / "features" / backbone / dataset / split_id / section / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "feature_cache.pt"
    features = [[1.0, 0.0] if label == 0 else [0.0, 1.0] for label in labels]
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "image_features": features,
                "image_labels": labels,
                "image_paths": [f"fake://{split_id}/{section}/{idx}.jpg" for idx in range(len(labels))],
                "split_name": section,
                "class_to_idx": {"class_0": 0, "class_1": 1},
                "backbone": backbone,
                "dataset": dataset,
                "feature_dim": 2,
                "normalize_features": True,
                "created_at": "2026-05-12T00:00:00+00:00",
                "source_script": "tests/test_run_cached_adapters.py",
                "metadata": {"uses_fake_features": True, "uses_fake_data": True},
            },
            handle,
        )
    summary_path = run_dir / "feature_extraction_summary.json"
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": str(split_path),
            "split_section": section,
            "image_count": len(labels),
            "feature_shape": [len(labels), 2],
            "feature_cache_path": str(cache_path),
            "run_dir": str(run_dir),
            "checkpoint_loaded": True,
            "is_paper_result": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "saves_predictions": False,
            "saves_logits": False,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(cache_path), "run_dir": str(run_dir)}


def write_text_cache(path: Path, *, dataset: str, backbone: str, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "text_features": [[1.0, 0.0], [0.0, 1.0]],
                "class_names": ["class_0", "class_1"],
                "class_to_idx": {"class_0": 0, "class_1": 1},
                "prompts": ["a satellite photo of class_0.", "a satellite photo of class_1."],
                "dataset": dataset,
                "backbone": backbone,
                "base_split": "base_seed1",
                "feature_dim": 2,
                "num_classes": 2,
                "normalize_features": True,
                "source_script": "tests/test_run_cached_adapters.py",
                "created_at": "2026-05-12T00:00:00+00:00",
                "dry_run": dry_run,
                "uses_fake_text_features": dry_run,
                "is_paper_result": False,
            },
            handle,
        )


def write_adapter_preflight_report(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    base_split_path: Path,
    shot_split_path: Path,
    shot: int,
    method_ready: bool,
) -> Path:
    path = root / "outputs" / "preflight" / "adapter_input" / "adapter_input_preflight_report.json"
    safe_write_json(
        path,
        {
            "is_valid": method_ready,
            "dataset": dataset,
            "backbone": backbone,
            "per_split_summary": {
                str(base_split_path): {
                    "sections": {"val": {"is_ready": True}, "test": {"is_ready": True}},
                    "val_ready_for_tuning_input": True,
                    "test_ready_for_evaluation_input": True,
                }
            },
            "per_method_input_summary": {
                "tip_adapter": {
                    "per_shot": {
                        str(shot_split_path): {
                            "method_input_ready": method_ready,
                            "shot": shot,
                            "expected_cache_entries": 2 * shot,
                            "actual_support_entries": 2 * shot,
                        }
                    }
                },
                "proto_adapter": {
                    "per_shot": {
                        str(shot_split_path): {
                            "method_input_ready": method_ready,
                            "shot": shot,
                            "expected_cache_entries": 2,
                            "actual_support_entries": 2 * shot,
                        }
                    }
                },
            },
        },
    )
    return path


def write_adapter_input_plan(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    shot_split_path: Path,
    shot: int,
    plan_ready: bool,
) -> Path:
    path = root / "outputs" / "preflight" / "adapter_input_plans" / "adapter_input_plan.json"
    rows = [
        {
            "dataset": dataset,
            "backbone": backbone,
            "seed": "seed1",
            "shot_split": str(shot_split_path),
            "shot": shot,
            "method": "tip_adapter",
            "support_entries": 2 * shot,
            "candidate_M": None,
            "is_ready": plan_ready,
            "skip_reason": "" if plan_ready else "preflight_method_input_not_ready",
            "expected_cache_entries": 2 * shot,
        },
        {
            "dataset": dataset,
            "backbone": backbone,
            "seed": "seed1",
            "shot_split": str(shot_split_path),
            "shot": shot,
            "method": "proto_adapter",
            "support_entries": 2 * shot,
            "candidate_M": None,
            "is_ready": plan_ready,
            "skip_reason": "" if plan_ready else "preflight_method_input_not_ready",
            "expected_cache_entries": 2,
        },
    ]
    safe_write_json(
        path,
        {
            "source_preflight_is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "seed": "seed1",
            "num_classes": 2,
            "feature_dim": 2,
            "rows": rows,
        },
    )
    return path


if __name__ == "__main__":
    unittest.main()
