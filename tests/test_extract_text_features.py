from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.check_text_feature_cache_preflight import run_text_feature_cache_preflight
from scripts.extract_text_features import run_text_feature_extraction
from src.utils.io import read_json, safe_write_json


class ExtractTextFeaturesTest(unittest.TestCase):
    def test_dry_run_writes_standalone_text_cache_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_inputs(root)

            summary_path, is_valid = run_text_feature_extraction(
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                preflight_report=inputs["preflight_report"],
                backbone_config=inputs["backbone_config"],
                method_config=inputs["method_config"],
                output_dir=root / "outputs" / "features",
                weights_path=None,
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                dry_run=True,
                command="pytest dry-run text extraction",
            )

            summary = read_json(summary_path)
            cache_path = Path(summary["text_feature_cache_path"])
            self.assertTrue(is_valid)
            self.assertTrue(summary["is_valid"])
            self.assertTrue(cache_path.exists())
            self.assertEqual(summary["feature_shape"], [3, 512])
            self.assertEqual(summary["num_classes"], 3)
            self.assertEqual(summary["feature_dim"], 512)
            self.assertFalse(summary["loads_model"])
            self.assertFalse(summary["extracts_text_features"])
            self.assertFalse(summary["computes_logits"])
            self.assertFalse(summary["computes_accuracy"])
            self.assertFalse(summary["evaluates_model"])
            self.assertFalse(summary["trains_model"])
            self.assertFalse(summary["saves_predictions"])
            self.assertFalse(summary["writes_results_raw"])
            self.assertFalse(summary["is_paper_result"])
            self.assertTrue(summary["dry_run"])
            self.assertTrue(summary["uses_fake_text_features"])
            self.assertEqual(cache_path.parent.parent, root / "outputs" / "features" / "eurosat" / "remoteclip_vit_b32" / "text")
            self.assertIn("text", cache_path.parts)
            self.assertNotEqual(cache_path.name, "feature_cache.pt")

            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
            self.assertIsInstance(cache["text_features"], torch.Tensor)
            self.assertEqual(list(cache["text_features"].shape), [3, 512])
            self.assertEqual(cache["class_names"], ["class_0", "class_1", "class_2"])
            self.assertEqual(cache["class_to_idx"], {"class_0": 0, "class_1": 1, "class_2": 2})
            self.assertEqual(cache["prompt_templates"], ["a satellite photo of {}."])
            self.assertFalse(cache["is_paper_result"])
            self.assertTrue(cache["dry_run"])
            self.assertTrue(cache["uses_fake_text_features"])
            self.assertFalse(cache["computes_logits"])
            self.assertFalse(cache["computes_accuracy"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_missing_class_names_fails_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_inputs(root, class_names=[])

            summary_path, is_valid = run_text_feature_extraction(
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                preflight_report=inputs["preflight_report"],
                backbone_config=inputs["backbone_config"],
                method_config=inputs["method_config"],
                output_dir=root / "outputs" / "features",
                weights_path=None,
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                dry_run=True,
                command="pytest missing class names",
            )

            summary = read_json(summary_path)
            self.assertFalse(is_valid)
            self.assertFalse(summary["is_valid"])
            self.assertIsNone(summary["text_feature_cache_path"])
            self.assertTrue(any("class_names" in error for error in summary["errors"]))
            self.assertFalse(summary["saves_text_feature_cache"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_missing_feature_dim_fails_without_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_inputs(root, expected_feature_dim=None)

            summary_path, is_valid = run_text_feature_extraction(
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                preflight_report=inputs["preflight_report"],
                backbone_config=inputs["backbone_config"],
                method_config=inputs["method_config"],
                output_dir=root / "outputs" / "features",
                weights_path=None,
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                dry_run=True,
                command="pytest missing feature dim",
            )

            summary = read_json(summary_path)
            self.assertFalse(is_valid)
            self.assertFalse(summary["is_valid"])
            self.assertIsNone(summary["text_feature_cache_path"])
            self.assertTrue(any("expected_feature_dim" in error for error in summary["errors"]))
            self.assertFalse(summary["saves_text_feature_cache"])
            self.assertFalse(summary["computes_logits"])
            self.assertFalse(summary["computes_accuracy"])

    def test_dry_run_output_path_is_discoverable_by_text_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = write_discoverable_inputs(root, dataset="aid", backbone="remoteclip_vit_b32")

            summary_path, is_valid = run_text_feature_extraction(
                dataset="aid",
                backbone="remoteclip_vit_b32",
                base_split=str(inputs["base_split_path"]),
                preflight_report=inputs["preflight_report"],
                backbone_config=inputs["backbone_config"],
                method_config=inputs["method_config"],
                output_dir=root / "outputs" / "features",
                weights_path=None,
                device="cpu",
                execution_env="local_wsl",
                run_mode="local_validation",
                dry_run=True,
                command="pytest discoverable dry-run text extraction",
            )

            summary = read_json(summary_path)
            cache_path = Path(summary["text_feature_cache_path"])
            self.assertTrue(is_valid)
            self.assertEqual(cache_path.parent.parent, root / "outputs" / "features" / "aid" / "remoteclip_vit_b32" / "text")
            self.assertFalse(summary["is_paper_result"])
            self.assertTrue(summary["dry_run"])
            self.assertTrue(summary["uses_fake_text_features"])

            report_path, preflight_valid = run_text_feature_cache_preflight(
                manifest_path=inputs["manifest_path"],
                dataset="aid",
                backbone="remoteclip_vit_b32",
                base_split=str(inputs["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest discover extracted text cache",
            )

            report = read_json(report_path)
            self.assertTrue(preflight_valid)
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(report["selected_text_feature_cache_path"], str(cache_path))
            self.assertEqual(
                report["proposed_text_feature_cache_path"],
                str(root / "outputs" / "features" / "aid" / "remoteclip_vit_b32" / "text" / "text_feature_cache.pt"),
            )
            self.assertFalse((root / "results" / "raw").exists())


def write_inputs(
    root: Path,
    *,
    class_names: list[str] | None = None,
    expected_feature_dim: int | None = 512,
) -> dict[str, Path]:
    if class_names is None:
        class_names = ["class_0", "class_1", "class_2"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    preflight_report = root / "preflight" / "text_feature_cache_preflight_report.json"
    preflight_data = {
        "is_valid": True,
        "text_feature_cache_exists": False,
        "text_feature_cache_ready": False,
        "dataset": "eurosat",
        "backbone": "remoteclip_vit_b32",
        "base_split": "base_seed1",
        "checked_base_split": {"split_id": "base_seed1", "seed": 1},
        "class_order_determinable": True,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "expected_feature_dim": expected_feature_dim,
        "prompt_templates": ["a satellite photo of {}."],
        "proposed_text_feature_cache_path": str(root / "unused" / "text_feature_cache.pt"),
    }
    safe_write_json(preflight_report, preflight_data)
    backbone_config = root / "configs" / "remoteclip_vit_b32.yaml"
    method_config = root / "configs" / "zero_shot_clip.yaml"
    safe_write_json(
        backbone_config,
        {
            "backbone": {
                "name": "remoteclip_vit_b32",
                "family": "remoteclip",
                "feature_dim": 512,
                "weights": None,
                "allow_download": False,
                "normalize_features": True,
            }
        },
    )
    safe_write_json(method_config, {"method": {"name": "zero_shot_clip", "prompt_templates": ["a satellite photo of {}."]}})
    return {
        "preflight_report": preflight_report,
        "backbone_config": backbone_config,
        "method_config": method_config,
    }


def write_discoverable_inputs(root: Path, *, dataset: str, backbone: str) -> dict[str, Path]:
    inputs = write_inputs(root)
    class_to_idx = {"class_0": 0, "class_1": 1, "class_2": 2}
    base_split_path = root / "splits" / dataset / "base_seed1.json"
    safe_write_json(
        base_split_path,
        {
            "dataset": dataset,
            "seed": 1,
            "num_classes": 3,
            "class_to_idx": class_to_idx,
            "train": [],
            "val": [],
            "test": [],
            "support": [],
        },
    )
    preflight_report = root / "preflight" / f"{dataset}_text_feature_cache_preflight_report.json"
    safe_write_json(
        preflight_report,
        {
            "is_valid": True,
            "text_feature_cache_exists": False,
            "text_feature_cache_ready": False,
            "dataset": dataset,
            "backbone": backbone,
            "base_split": str(base_split_path),
            "checked_base_split": {"split_id": "base_seed1", "seed": 1},
            "class_order_determinable": True,
            "class_names": ["class_0", "class_1", "class_2"],
            "class_to_idx": class_to_idx,
            "num_classes": 3,
            "expected_feature_dim": 512,
            "prompt_templates": ["a satellite photo of {}."],
            "proposed_text_feature_cache_path": str(
                root / "outputs" / "features" / dataset / backbone / "text" / "text_feature_cache.pt"
            ),
        },
    )
    entries = [
        write_image_summary(
            root,
            dataset=dataset,
            backbone=backbone,
            split_path=base_split_path,
            section=section,
            feature_dim=512,
        )
        for section in ["train", "val", "test"]
    ]
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    return {
        **inputs,
        "preflight_report": preflight_report,
        "base_split_path": base_split_path,
        "manifest_path": manifest_path,
    }


def write_image_summary(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    split_path: Path,
    section: str,
    feature_dim: int,
) -> dict[str, str]:
    run_dir = root / "outputs" / "features" / dataset / backbone / section / "20260518T000000"
    summary_path = run_dir / "feature_extraction_summary.json"
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": str(split_path),
            "split_section": section,
            "image_count": 3,
            "feature_shape": [3, feature_dim],
            "feature_dim": feature_dim,
            "feature_cache_path": str(run_dir / "feature_cache.pt"),
            "run_dir": str(run_dir),
            "is_paper_result": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "extracts_text_features": False,
            "saves_predictions": False,
            "saves_logits": False,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(run_dir / "feature_cache.pt"), "run_dir": str(run_dir)}


if __name__ == "__main__":
    unittest.main()
