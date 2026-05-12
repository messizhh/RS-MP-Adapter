from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.check_zero_shot_eval_preflight import run_zero_shot_eval_preflight
from src.utils.io import read_json, safe_write_json


class ZeroShotEvalPreflightTest(unittest.TestCase):
    def test_text_features_present_with_valid_shape_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, base_split_path = write_fake_case(root, text_mode="valid")

            report_path, is_valid = run_zero_shot_eval_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(base_split_path),
                output_dir=root / "outputs" / "preflight" / "zero_shot_eval",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest zero-shot eval preflight valid",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["zero_shot_input_ready"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["feature_dim"], 4)
            self.assertEqual(report["num_classes"], 3)
            self.assertTrue(report["text_feature_summary"]["has_valid_text_features"])
            self.assertEqual(set(report["text_feature_summary"]["valid_text_feature_sections"]), {"train", "val", "test"})
            self.assertTrue(report["val_ready_for_eval_input"])
            self.assertTrue(report["test_ready_for_eval_input"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["writes_results_raw"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_missing_text_features_marks_not_ready_with_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, base_split_path = write_fake_case(root, text_mode="missing")

            report_path, is_valid = run_zero_shot_eval_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(base_split_path),
                output_dir=root / "outputs" / "preflight" / "zero_shot_eval",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest zero-shot eval preflight missing text",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertFalse(report["zero_shot_input_ready"])
            self.assertTrue(any("text_features" in error for error in report["errors"]))
            self.assertTrue(any("Generate or attach text_features" in item for item in report["recommendations"]))
            self.assertFalse(report["text_feature_summary"]["has_valid_text_features"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_bad_text_feature_shape_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, base_split_path = write_fake_case(root, text_mode="bad_shape")

            report_path, is_valid = run_zero_shot_eval_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(base_split_path),
                output_dir=root / "outputs" / "preflight" / "zero_shot_eval",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest zero-shot eval preflight bad text shape",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["zero_shot_input_ready"])
            self.assertTrue(any("text_features shape" in error for error in report["errors"]))
            self.assertFalse(report["text_feature_summary"]["by_section"]["val"]["shape_valid"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, base_split_path = write_fake_case(root, text_mode="valid")

            with self.assertRaisesRegex(ValueError, "results/raw"):
                run_zero_shot_eval_preflight(
                    manifest_path=manifest_path,
                    dataset="eurosat",
                    backbone="remoteclip_vit_b32",
                    base_split=str(base_split_path),
                    output_dir=root / "results" / "raw" / "zero_shot_eval",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    command="pytest zero-shot reject results raw",
                )


def write_fake_case(root: Path, *, text_mode: str) -> tuple[Path, Path]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 3
    feature_dim = 4
    base_split_path = write_base_split(root, dataset, num_classes)
    entries = []
    for section in ["train", "val", "test"]:
        labels = [idx % num_classes for idx in range(6)]
        bad_text_shape = text_mode == "bad_shape" and section == "val"
        include_text = text_mode != "missing"
        entries.append(
            write_summary_and_cache(
                root,
                dataset=dataset,
                backbone=backbone,
                base_split_path=base_split_path,
                section=section,
                labels=labels,
                feature_dim=feature_dim,
                num_classes=num_classes,
                include_text=include_text,
                bad_text_shape=bad_text_shape,
            )
        )
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    return manifest_path, base_split_path


def write_base_split(root: Path, dataset: str, num_classes: int) -> Path:
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    data = {
        "dataset": dataset,
        "seed": 1,
        "shot": None,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
    }
    for section in ["train", "val", "test"]:
        data[section] = [
            {"class_name": f"class_{idx % num_classes}", "label": idx % num_classes, "path": f"{section}/{idx}.jpg"}
            for idx in range(6)
        ]
    data["support"] = []
    path = root / "splits" / dataset / "base_seed1.json"
    safe_write_json(path, data)
    return path


def write_summary_and_cache(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    base_split_path: Path,
    section: str,
    labels: list[int],
    feature_dim: int,
    num_classes: int,
    include_text: bool,
    bad_text_shape: bool,
) -> dict[str, str]:
    run_dir = root / "features" / backbone / dataset / "base_seed1" / section / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "feature_cache.pt"
    summary_path = run_dir / "feature_extraction_summary.json"
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    cache = {
        "image_features": [[float(idx + 1) for _ in range(feature_dim)] for idx in range(len(labels))],
        "image_labels": labels,
        "image_paths": [f"fake://{section}/{idx}.jpg" for idx in range(len(labels))],
        "split_name": section,
        "class_to_idx": class_to_idx,
        "backbone": backbone,
        "dataset": dataset,
        "feature_dim": feature_dim,
        "normalize_features": True,
        "created_at": "2026-05-12T00:00:00+00:00",
        "source_script": "tests/test_zero_shot_eval_preflight.py",
        "metadata": {"prompt_templates": ["a satellite photo of {}."]},
    }
    if include_text:
        text_rows = num_classes - 1 if bad_text_shape else num_classes
        cache["text_features"] = [[float(label + 1) for _ in range(feature_dim)] for label in range(text_rows)]
        cache["text_prompts"] = [f"a satellite photo of class_{idx}." for idx in range(num_classes)]
        cache["text_class_names"] = [f"class_{idx}" for idx in range(num_classes)]
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle)
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": str(base_split_path),
            "split_section": section,
            "image_count": len(labels),
            "feature_shape": [len(labels), feature_dim],
            "text_feature_shape": [num_classes, feature_dim] if include_text and not bad_text_shape else None,
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


if __name__ == "__main__":
    unittest.main()
