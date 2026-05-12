from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_result_run_preflight import run_result_run_preflight
from src.utils.io import read_json, safe_write_json


class ResultRunPreflightTest(unittest.TestCase):
    def test_valid_fake_zero_shot_run_dir_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest valid result run preflight",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["errors"], [])
            self.assertTrue(report["checked_files"]["metrics.json"]["exists"])
            self.assertTrue(report["checked_files"]["metadata.json"]["exists"])
            self.assertTrue(report["checked_files"]["config"]["exists"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["modifies_results"])
            self.assertFalse(report["deletes_results"])
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["paper_filtering_summary"]["metadata_is_paper_result"])
            self.assertEqual(report["metrics_summary"]["top1_acc_by_split"], {"test": 0.6, "val": 0.5})
            self.assertIsNone(report["metrics_summary"]["shot"])

    def test_zero_shot_cache_entries_num_classes_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, cache_entries=2)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest zero shot cache entries",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["errors"], [])

    def test_proto_adapter_shot_2_cache_entries_num_classes_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, method="proto_adapter", shot=2, cache_entries=2)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="proto_adapter",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest proto adapter cache entries",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["metrics_summary"]["shot"], 2)

    def test_tip_adapter_shot_2_cache_entries_num_classes_times_shot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, method="tip_adapter", shot=2, cache_entries=4)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="tip_adapter",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest tip adapter cache entries",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["metrics_summary"]["shot"], 2)

    def test_tip_adapter_shot_2_cache_entries_num_classes_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, method="tip_adapter", shot=2, cache_entries=2)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="tip_adapter",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest bad tip adapter cache entries",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertTrue(any("tip_adapter cache_entries must equal num_classes * shot" in error for error in report["errors"]))

    def test_tip_adapter_missing_shot_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, method="tip_adapter", shot=None, cache_entries=2)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="tip_adapter",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest missing tip adapter shot",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertTrue(any("tip_adapter shot must be a positive integer" in error for error in report["errors"]))

    def test_missing_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root)
            (run_dir / "metadata.json").unlink()

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest missing metadata",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(any("metadata.json" in error for error in report["errors"]))

    def test_local_validation_paper_result_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, is_paper_result=True, eligible_for_paper_tables=True)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest paper flag local validation",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertTrue(any("is_paper_result=true" in error for error in report["errors"]))
            self.assertTrue(any("eligible_for_paper_tables=true" in error for error in report["errors"]))

    def test_invalid_top1_outside_unit_interval_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, top1_val=1.2)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest bad top1",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertTrue(any("top1_acc_by_split[val]" in error for error in report["errors"]))

    def test_confusion_matrix_wrong_shape_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root, bad_confusion_shape=True)

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest bad confusion",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertTrue(any("confusion_matrix" in error for error in report["errors"]))

    def test_checker_does_not_modify_existing_metrics_or_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_zero_shot_run(root)
            metrics_before = (run_dir / "metrics.json").read_bytes()
            metadata_before = (run_dir / "metadata.json").read_bytes()

            report_path, is_valid = run_result_run_preflight(
                run_dir=run_dir,
                expected_method="zero_shot",
                expected_dataset="eurosat",
                expected_backbone="remoteclip_vit_b32",
                output_dir=root / "outputs" / "preflight" / "result_runs",
                execution_env="remote_server",
                run_mode="local_validation",
                command="pytest readonly result checker",
            )

            self.assertTrue(is_valid)
            self.assertTrue(report_path.exists())
            self.assertEqual((run_dir / "metrics.json").read_bytes(), metrics_before)
            self.assertEqual((run_dir / "metadata.json").read_bytes(), metadata_before)


def write_fake_zero_shot_run(
    root: Path,
    *,
    method: str = "zero_shot",
    shot: int | None = None,
    cache_entries: int = 2,
    is_paper_result: bool = False,
    eligible_for_paper_tables: bool = False,
    top1_val: float = 0.5,
    top1_test: float = 0.6,
    bad_confusion_shape: bool = False,
) -> Path:
    shot_dir = f"shot_{shot}" if shot is not None else "seed_1"
    if method == "zero_shot":
        run_dir = root / "results" / "raw" / "eurosat" / "remoteclip_vit_b32" / method / "seed_1" / "fake_run"
    else:
        run_dir = root / "results" / "raw" / "eurosat" / "remoteclip_vit_b32" / method / shot_dir / "seed_1" / "fake_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(f"method:\n  name: {method}\n", encoding="utf-8")
    (run_dir / "log.txt").write_text("Experiment log initialized.\nExperiment finished.\n", encoding="utf-8")
    metadata = base_metadata(run_dir, method=method, shot=shot, is_paper_result=is_paper_result, eligible_for_paper_tables=eligible_for_paper_tables)
    metrics = base_metrics(
        run_dir,
        method=method,
        shot=shot,
        cache_entries=cache_entries,
        is_paper_result=is_paper_result,
        eligible_for_paper_tables=eligible_for_paper_tables,
        top1_val=top1_val,
        top1_test=top1_test,
        bad_confusion_shape=bad_confusion_shape,
    )
    safe_write_json(run_dir / "metadata.json", metadata)
    safe_write_json(run_dir / "metrics.json", metrics)
    return run_dir


def base_metadata(
    run_dir: Path,
    *,
    method: str,
    shot: int | None,
    is_paper_result: bool,
    eligible_for_paper_tables: bool,
) -> dict[str, object]:
    return {
        "run_id": "fake_run",
        "method": method,
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "seed": 1,
        "execution_env": "remote_server",
        "run_mode": "local_validation",
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": "cpu",
        "git_commit": "abc123",
        "python_version": "3.12",
        "pytorch_version": "2.0",
        "cuda_version": "not_available",
        "gpu_name": "CPU",
        "host_name": "test-host",
        "command": "pytest fake zero-shot",
        "config_path": f"configs/methods/{method}.yaml",
        "config_snapshot_path": str(run_dir / "config.yaml"),
        "shot": shot,
        "server_job_id": None,
        "base_split": "base_seed1",
        "shot_split": f"shot_{shot}_seed1" if shot is not None else None,
        "eval_splits": ["val", "test"],
        "split_path": f"shot_{shot}_seed1" if shot is not None else "base_seed1",
        "support_cache_path": f"features/support/shot_{shot}/feature_cache.pt" if shot is not None else None,
        "image_cache_paths": {"val": "features/val/feature_cache.pt", "test": "features/test/feature_cache.pt"},
        "text_feature_cache_path": "features/text/text_feature_cache.pt",
        "feature_dim": 512,
        "num_classes": 2,
        "start_time": "2026-05-12T00:00:00+00:00",
        "end_time": "2026-05-12T00:01:00+00:00",
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(run_dir / "log.txt"),
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": False,
        "writes_results_raw": True,
    }


def base_metrics(
    run_dir: Path,
    *,
    method: str,
    shot: int | None,
    cache_entries: int,
    is_paper_result: bool,
    eligible_for_paper_tables: bool,
    top1_val: float,
    top1_test: float,
    bad_confusion_shape: bool,
) -> dict[str, object]:
    per_split = {
        "val": split_metrics(top1_val, bad_confusion_shape=bad_confusion_shape),
        "test": split_metrics(top1_test, bad_confusion_shape=False),
    }
    return {
        "run_id": "fake_run",
        "method": method,
        "backbone": "remoteclip_vit_b32",
        "dataset": "eurosat",
        "shot": shot,
        "seed": 1,
        "execution_env": "remote_server",
        "run_mode": "local_validation",
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": "cpu",
        "base_split": "base_seed1",
        "shot_split": f"shot_{shot}_seed1" if shot is not None else None,
        "eval_splits": ["val", "test"],
        "support_cache_path": f"features/support/shot_{shot}/feature_cache.pt" if shot is not None else None,
        "image_cache_paths": {"val": "features/val/feature_cache.pt", "test": "features/test/feature_cache.pt"},
        "text_feature_cache_path": "features/text/text_feature_cache.pt",
        "feature_dim": 512,
        "num_classes": 2,
        "top1_acc": top1_test,
        "top1_acc_by_split": {"val": top1_val, "test": top1_test},
        "per_split": per_split,
        "num_samples": 4,
        "cache_entries": cache_entries,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "inference_time_sec": 0.1,
        "images_per_second": 40.0,
        "gpu_memory_mb": None,
        "uses_fake_data": True,
        "uses_fake_features": True,
        "fake_or_dry_run": True,
        "config_path": f"configs/methods/{method}.yaml",
        "config_snapshot_path": str(run_dir / "config.yaml"),
        "split_path": f"shot_{shot}_seed1" if shot is not None else "base_seed1",
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(run_dir / "log.txt"),
        "prediction_path": "",
        "checkpoint_path": None,
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": False,
        "writes_results_raw": True,
        "start_time": "2026-05-12T00:00:00+00:00",
        "end_time": "2026-05-12T00:01:00+00:00",
    }


def split_metrics(top1_acc: float, *, bad_confusion_shape: bool) -> dict[str, object]:
    return {
        "top1_acc": top1_acc,
        "num_samples": 2,
        "num_classes": 2,
        "per_class_acc": [
            {"class_name": "class_0", "class_idx": 0, "num_samples": 1, "num_correct": 1, "accuracy": 1.0},
            {"class_name": "class_1", "class_idx": 1, "num_samples": 1, "num_correct": 0, "accuracy": 0.0},
        ],
        "confusion_matrix": [[1, 0]] if bad_confusion_shape else [[1, 0], [1, 0]],
        "inference_time_sec": 0.05,
        "images_per_second": 40.0,
    }


if __name__ == "__main__":
    unittest.main()
