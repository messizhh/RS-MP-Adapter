from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.run_zero_shot import run_cached_zero_shot_evaluation
from src.utils.io import read_json, safe_write_json


class RunZeroShotCachedTest(unittest.TestCase):
    def test_fake_cached_zero_shot_writes_metrics_without_predictions_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_case(root, text_dry_run=True, preflight_ready=True)

            result = run_cached_zero_shot_evaluation(
                config={"method": {"name": "zero_shot_clip"}},
                config_path="configs/methods/zero_shot_clip.yaml",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                manifest_path=case["manifest_path"],
                text_feature_cache_path=case["text_cache_path"],
                eval_splits=["val", "test"],
                output_dir=root / "results" / "raw",
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                preflight_report=case["preflight_report"],
                dry_run=True,
                max_samples=None,
                seed=1,
                save_predictions=False,
                allow_paper_result=False,
                skip_preflight_check=False,
                command="pytest cached zero shot",
            )

            metrics = read_json(result["metrics_path"])
            metadata = read_json(result["metadata_path"])
            self.assertEqual(metrics["method"], "zero_shot")
            self.assertEqual(metrics["top1_acc_by_split"], {"test": 1.0, "val": 1.0})
            self.assertEqual(metrics["feature_dim"], 2)
            self.assertEqual(metrics["num_classes"], 2)
            self.assertFalse(metrics["is_paper_result"])
            self.assertFalse(metadata["is_paper_result"])
            self.assertTrue(metrics["computes_logits"])
            self.assertTrue(metrics["computes_accuracy"])
            self.assertTrue(metrics["evaluates_model"])
            self.assertFalse(metrics["trains_model"])
            self.assertFalse(metrics["extracts_features"])
            self.assertFalse(metrics["loads_model"])
            self.assertFalse(metrics["saves_predictions"])
            self.assertTrue(metrics["writes_results_raw"])
            self.assertEqual(metrics["text_feature_cache_path"], str(case["text_cache_path"]))
            self.assertFalse((Path(result["run_dir"]) / "predictions.csv").exists())
            self.assertTrue((Path(result["run_dir"]) / "config.yaml").exists())
            self.assertTrue((Path(result["run_dir"]) / "log.txt").exists())

    def test_save_predictions_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_case(root, text_dry_run=True, preflight_ready=True)

            result = run_cached_zero_shot_evaluation(
                config={"method": {"name": "zero_shot_clip"}},
                config_path="configs/methods/zero_shot_clip.yaml",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                manifest_path=case["manifest_path"],
                text_feature_cache_path=case["text_cache_path"],
                eval_splits=["val"],
                output_dir=root / "results" / "raw",
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                preflight_report=case["preflight_report"],
                dry_run=True,
                max_samples=None,
                seed=1,
                save_predictions=True,
                allow_paper_result=False,
                skip_preflight_check=False,
                command="pytest cached zero shot save predictions",
            )

            metrics = read_json(result["metrics_path"])
            self.assertTrue(metrics["saves_predictions"])
            self.assertTrue(Path(metrics["prediction_path"]).exists())

    def test_fake_text_cache_is_rejected_for_non_dry_run_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_case(root, text_dry_run=True, preflight_ready=True)

            with self.assertRaisesRegex(ValueError, "dry-run/fake text cache"):
                run_cached_zero_shot_evaluation(
                    config={"method": {"name": "zero_shot_clip"}},
                    config_path="configs/methods/zero_shot_clip.yaml",
                    dataset="eurosat",
                    backbone="remoteclip_vit_b32",
                    base_split=str(case["base_split_path"]),
                    manifest_path=case["manifest_path"],
                    text_feature_cache_path=case["text_cache_path"],
                    eval_splits=["val"],
                    output_dir=root / "results" / "raw",
                    device="cpu",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    preflight_report=case["preflight_report"],
                    dry_run=False,
                    max_samples=None,
                    seed=1,
                    save_predictions=False,
                    allow_paper_result=False,
                    skip_preflight_check=False,
                    command="pytest reject fake text",
                )

    def test_preflight_not_ready_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_case(root, text_dry_run=False, preflight_ready=False)

            with self.assertRaisesRegex(ValueError, "preflight report is not ready"):
                run_cached_zero_shot_evaluation(
                    config={"method": {"name": "zero_shot_clip"}},
                    config_path="configs/methods/zero_shot_clip.yaml",
                    dataset="eurosat",
                    backbone="remoteclip_vit_b32",
                    base_split=str(case["base_split_path"]),
                    manifest_path=case["manifest_path"],
                    text_feature_cache_path=case["text_cache_path"],
                    eval_splits=["val"],
                    output_dir=root / "results" / "raw",
                    device="cpu",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    preflight_report=case["preflight_report"],
                    dry_run=False,
                    max_samples=None,
                    seed=1,
                    save_predictions=False,
                    allow_paper_result=False,
                    skip_preflight_check=False,
                    command="pytest reject preflight",
                )

    def test_run_directory_is_unique_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_cached_case(root, text_dry_run=True, preflight_ready=True)
            kwargs = {
                "config": {"method": {"name": "zero_shot_clip"}},
                "config_path": "configs/methods/zero_shot_clip.yaml",
                "dataset": "eurosat",
                "backbone": "remoteclip_vit_b32",
                "base_split": str(case["base_split_path"]),
                "manifest_path": case["manifest_path"],
                "text_feature_cache_path": case["text_cache_path"],
                "eval_splits": ["val"],
                "output_dir": root / "results" / "raw",
                "device": "cpu",
                "execution_env": "local_wsl",
                "run_mode": "local_validation",
                "preflight_report": case["preflight_report"],
                "dry_run": True,
                "max_samples": None,
                "seed": 1,
                "save_predictions": False,
                "allow_paper_result": False,
                "skip_preflight_check": False,
                "command": "pytest unique run dir",
            }

            first = run_cached_zero_shot_evaluation(**kwargs)
            second = run_cached_zero_shot_evaluation(**kwargs)

            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertTrue(Path(first["metrics_path"]).exists())
            self.assertTrue(Path(second["metrics_path"]).exists())


def write_cached_case(root: Path, *, text_dry_run: bool, preflight_ready: bool) -> dict[str, Path]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    base_split_path = write_base_split(root, dataset=dataset)
    entries = []
    for split in ["train", "val", "test"]:
        entries.append(write_image_cache(root, dataset=dataset, backbone=backbone, split=split, base_split_path=base_split_path))
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    text_cache_path = root / "features" / backbone / dataset / "base_seed1" / "text" / "20260512T140232" / "text_feature_cache.pt"
    write_text_cache(text_cache_path, dataset=dataset, backbone=backbone, dry_run=text_dry_run)
    preflight_report = root / "preflight" / "zero_shot_eval_preflight_report.json"
    safe_write_json(
        preflight_report,
        {
            "is_valid": preflight_ready,
            "zero_shot_input_ready": preflight_ready,
            "standalone_text_feature_cache_ready": preflight_ready,
            "text_feature_source": "standalone_cache",
        },
    )
    return {
        "base_split_path": base_split_path,
        "manifest_path": manifest_path,
        "text_cache_path": text_cache_path,
        "preflight_report": preflight_report,
    }


def write_base_split(root: Path, *, dataset: str) -> Path:
    path = root / "splits" / dataset / "base_seed1.json"
    class_to_idx = {"class_0": 0, "class_1": 1}
    rows = [
        {"class_name": "class_0", "label": 0, "path": "class_0/0.jpg"},
        {"class_name": "class_1", "label": 1, "path": "class_1/1.jpg"},
    ]
    safe_write_json(
        path,
        {
            "dataset": dataset,
            "seed": 1,
            "shot": None,
            "class_to_idx": class_to_idx,
            "num_classes": 2,
            "train": rows,
            "val": rows,
            "test": rows,
            "support": [],
        },
    )
    return path


def write_image_cache(root: Path, *, dataset: str, backbone: str, split: str, base_split_path: Path) -> dict[str, str]:
    run_dir = root / "features" / backbone / dataset / "base_seed1" / split / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "feature_cache.pt"
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "image_features": [[1.0, 0.0], [0.0, 1.0]],
                "image_labels": [0, 1],
                "image_paths": [f"fake://{split}/0.jpg", f"fake://{split}/1.jpg"],
                "split_name": split,
                "class_to_idx": {"class_0": 0, "class_1": 1},
                "backbone": backbone,
                "dataset": dataset,
                "feature_dim": 2,
                "normalize_features": True,
                "created_at": "2026-05-12T00:00:00+00:00",
                "source_script": "tests/test_run_zero_shot_cached.py",
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
            "split_path": str(base_split_path),
            "split_section": split,
            "image_count": 2,
            "feature_shape": [2, 2],
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
                "prompt_templates": ["a satellite photo of {}."],
                "dataset": dataset,
                "backbone": backbone,
                "base_split": "base_seed1",
                "feature_dim": 2,
                "num_classes": 2,
                "normalize_features": True,
                "source_script": "tests/test_run_zero_shot_cached.py",
                "created_at": "2026-05-12T14:02:32+00:00",
                "git_commit": "abc123",
                "execution_env": "local_wsl",
                "run_mode": "local_validation",
                "dry_run": dry_run,
                "uses_fake_text_features": dry_run,
                "is_paper_result": False,
            },
            handle,
        )


if __name__ == "__main__":
    unittest.main()
