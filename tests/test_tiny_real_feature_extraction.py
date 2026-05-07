from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from scripts.run_tiny_real_feature_extraction import run_tiny_real_feature_extraction
from src.features.feature_cache import load_feature_cache
from src.utils.io import read_json


class TinyRealFeatureExtractionTest(unittest.TestCase):
    def test_split_driven_tiny_runner_writes_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path, config_path, output_dir = make_backbone_inputs(base)
            dataset_root = base / "dataset"
            image_a = dataset_root / "class_a" / "a.png"
            image_b = dataset_root / "class_b" / "b.png"
            make_image(image_a)
            make_image(image_b)
            split_path = base / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "dataset": "eurosat",
                        "class_to_idx": {"class_a": 0, "class_b": 1},
                        "support": [
                            {"path": "class_a/a.png", "label": 0, "class_name": "class_a"},
                            {"path": "class_b/b.png", "label": 1, "class_name": "class_b"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patched_remoteclip_modules(checkpoint={"state_dict": {"visual.proj": torch.ones(1)}}):
                report_path, is_valid = run_tiny_real_feature_extraction(
                    backbone_config_path=config_path,
                    expected_backbone="remoteclip_vit_b32",
                    weights_path_override=str(weights_path),
                    dataset="eurosat",
                    split_path=split_path,
                    split_section="support",
                    image_list_path=None,
                    dataset_root=dataset_root,
                    max_images=2,
                    output_dir=output_dir,
                    execution_env="local_wsl",
                    run_mode="tiny_subset",
                    device="cpu",
                    command="pytest-controlled-command",
                )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["run_mode"], "tiny_subset")
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["eligible_for_paper_tables"])
            self.assertTrue(report["checkpoint_loaded"])
            self.assertTrue(report["reads_image_pixels"])
            self.assertTrue(report["extracts_features"])
            self.assertFalse(report["extracts_text_features"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["saves_logits"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertEqual(report["image_count"], 2)
            self.assertEqual(report["max_images"], 2)
            self.assertEqual(report["split_path"], str(split_path))
            self.assertIsNone(report["image_list_path"])
            self.assertEqual(report["weights_source"], "cli_override")
            self.assertIsNone(report["open_clip_initial_pretrained"])
            self.assertTrue(report["open_clip_initialization_warning_expected"])
            self.assertTrue(report["checkpoint_load_happened_after_model_init"])
            self.assertTrue(report["final_weights_loaded_from_checkpoint"])
            self.assertEqual(report["final_weight_source"], "cli_override_checkpoint")
            self.assertEqual(report["final_checkpoint_load_status"], "loaded_strictly_matching_keys")
            self.assertEqual(report["feature_shape"], [2, 512])
            self.assertNotIn("outputs/features/", report["feature_cache_path"])

            cache = load_feature_cache(report["feature_cache_path"])
            self.assertEqual(tuple(cache.image_features.shape), (2, 512))
            self.assertIsNone(cache.text_features)
            self.assertIsNone(cache.text_prompts)
            self.assertEqual(cache.image_labels.tolist(), [0, 1])
            for key in (
                "execution_env",
                "run_mode",
                "is_paper_result",
                "eligible_for_paper_tables",
                "dataset",
                "backbone",
                "image_count",
                "max_images",
                "split_path",
                "weights_source",
                "checkpoint_loaded",
                "final_weights_loaded_from_checkpoint",
                "final_checkpoint_load_status",
                "feature_shape",
                "feature_norm_stats",
                "command",
                "git_commit",
            ):
                self.assertIn(key, cache.metadata)

    def test_image_list_is_explicit_and_tiny_output_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path, config_path, _ = make_backbone_inputs(base)
            image_path = base / "listed.png"
            image_list_path = base / "images.txt"
            make_image(image_path)
            image_list_path.write_text(f"{image_path}\n", encoding="utf-8")

            with patched_remoteclip_modules(checkpoint={"state_dict": {"visual.proj": torch.ones(1)}}):
                report_path, is_valid = run_tiny_real_feature_extraction(
                    backbone_config_path=config_path,
                    expected_backbone="remoteclip_vit_b32",
                    weights_path_override=str(weights_path),
                    dataset="aid",
                    split_path=None,
                    split_section="support",
                    image_list_path=image_list_path,
                    dataset_root=None,
                    max_images=1,
                    output_dir=base / "outputs" / "features_tiny_preflight",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    device="cpu",
                )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(report["image_list_path"], str(image_list_path))
            self.assertIn("outputs/features_tiny_preflight", report["feature_cache_path"])

    def test_rejects_non_tiny_run_modes_and_formal_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path, config_path, _ = make_backbone_inputs(base)
            image_path = base / "listed.png"
            image_list_path = base / "images.txt"
            make_image(image_path)
            image_list_path.write_text(f"{image_path}\n", encoding="utf-8")

            report_path, is_valid = run_tiny_real_feature_extraction(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                dataset="aid",
                split_path=None,
                split_section="support",
                image_list_path=image_list_path,
                dataset_root=None,
                max_images=1,
                output_dir=base / "outputs" / "features",
                execution_env="local_wsl",
                run_mode="server_full",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["saves_feature_cache"])
            self.assertIn("--run-mode", " ".join(report["errors"]))
            self.assertIn("outputs/features", " ".join(report["errors"]))

    def test_rejects_missing_explicit_input_and_max_images_above_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path, config_path, output_dir = make_backbone_inputs(base)

            report_path, is_valid = run_tiny_real_feature_extraction(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                dataset="nwpu_resisc45",
                split_path=None,
                split_section="support",
                image_list_path=None,
                dataset_root=None,
                max_images=33,
                output_dir=output_dir,
                execution_env="local_wsl",
                run_mode="tiny_subset",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["saves_feature_cache"])
            self.assertIn("--max-images", " ".join(report["errors"]))
            self.assertIn("either --split or --image-list", " ".join(report["errors"]))


def make_backbone_inputs(base: Path) -> tuple[Path, Path, Path]:
    weights_path = base / "remoteclip.pt"
    weights_path.write_bytes(b"placeholder\n")
    config_path = base / "backbone.yaml"
    output_dir = base / "outputs" / "preflight"
    data = {
        "backbone": {
            "name": "remoteclip_vit_b32",
            "family": "remoteclip",
            "feature_dim": 512,
            "image_size": 224,
            "weights": None,
            "allow_download": False,
            "normalize_features": True,
            "preprocess": {"resize": 224, "center_crop": 224, "normalize": True},
        }
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return weights_path, config_path, output_dir


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(30, 20, 10)).save(path, format="PNG")


class FakeOpenClipModel:
    def load_state_dict(self, state_dict: dict[str, object], strict: bool = False) -> SimpleNamespace:
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones((image_tensor.shape[0], 512), dtype=torch.float32, device=image_tensor.device)

    def eval(self) -> "FakeOpenClipModel":
        return self


@contextmanager
def patched_remoteclip_modules(*, checkpoint: dict[str, object]):
    fake_open_clip = SimpleNamespace(
        __version__="fake-open-clip",
        create_model=lambda name, pretrained=None, device="cpu": FakeOpenClipModel(),
    )
    with patch("torch.load", lambda path, map_location=None: checkpoint), patch.dict(
        sys.modules, {"open_clip": fake_open_clip}
    ):
        yield


if __name__ == "__main__":
    unittest.main()
