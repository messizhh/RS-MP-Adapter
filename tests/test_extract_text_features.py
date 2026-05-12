from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import torch

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


if __name__ == "__main__":
    unittest.main()
