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

from scripts.check_backbone_feature_preflight import run_backbone_feature_preflight
from src.utils.io import read_json


class BackboneFeaturePreflightTest(unittest.TestCase):
    def test_remoteclip_single_image_feature_preflight_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_path = base / "sample.png"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_path)
            config_path, output_dir = make_inputs(base)
            checkpoint = {"state_dict": {"visual.proj": torch.ones(1), "text_projection": torch.ones(1)}}

            with patched_remoteclip_modules(checkpoint=checkpoint):
                report_path, is_valid = run_backbone_feature_preflight(
                    backbone_config_path=config_path,
                    expected_backbone="remoteclip_vit_b32",
                    weights_path_override=str(weights_path),
                    image_paths=[image_path],
                    output_dir=output_dir,
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    device="cpu",
                )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["loads_model"])
            self.assertTrue(report["checkpoint_loaded"])
            self.assertIsNone(report["open_clip_initial_pretrained"])
            self.assertTrue(report["open_clip_initialization_warning_expected"])
            self.assertTrue(report["checkpoint_load_happened_after_model_init"])
            self.assertTrue(report["final_weights_loaded_from_checkpoint"])
            self.assertEqual(report["final_weight_source"], "cli_override_checkpoint")
            self.assertEqual(report["final_checkpoint_load_status"], "loaded_strictly_matching_keys")
            self.assertTrue(report["reads_image_pixels"])
            self.assertTrue(report["extracts_features"])
            self.assertFalse(report["extracts_text_features"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["saves_feature_cache"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["saves_logits"])
            self.assertFalse(report["downloads_weights"])
            self.assertFalse(report["is_paper_result"])
            self.assertEqual(report["execution_env"], "local_wsl")
            self.assertEqual(report["run_mode"], "local_validation")
            self.assertEqual(report["image_count"], 1)
            self.assertEqual(report["feature_shape"], [1, 512])
            self.assertGreater(report["feature_norm"], 0.0)
            self.assertTrue(report["feature_is_finite"])

    def test_missing_weights_fails_before_model_or_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            image_path = base / "sample.png"
            make_image(image_path)
            config_path, output_dir = make_inputs(base)

            report_path, is_valid = run_backbone_feature_preflight(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(base / "missing.pt"),
                image_paths=[image_path],
                output_dir=output_dir,
                execution_env="local_wsl",
                run_mode="local_validation",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["reads_image_pixels"])
            self.assertFalse(report["extracts_features"])
            self.assertIn("does not exist", report["errors"][0])

    def test_multiple_image_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_a = base / "a.png"
            image_b = base / "b.png"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_a)
            make_image(image_b)
            config_path, output_dir = make_inputs(base)

            report_path, is_valid = run_backbone_feature_preflight(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                image_paths=[image_a, image_b],
                output_dir=output_dir,
                execution_env="local_wsl",
                run_mode="local_validation",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertEqual(report["image_count"], 2)
            self.assertIn("exactly one", report["errors"][0])

    def test_allow_download_true_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_path = base / "sample.png"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_path)
            config_path, output_dir = make_inputs(base, allow_download=True)

            report_path, is_valid = run_backbone_feature_preflight(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                image_paths=[image_path],
                output_dir=output_dir,
                execution_env="local_wsl",
                run_mode="local_validation",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertIn("allow_download", report["errors"][0])


def make_inputs(base: Path, allow_download: bool = False) -> tuple[Path, Path]:
    config_path = base / "backbone.yaml"
    output_dir = base / "reports"
    data = {
        "backbone": {
            "name": "remoteclip_vit_b32",
            "family": "remoteclip",
            "feature_dim": 512,
            "image_size": 224,
            "weights": None,
            "allow_download": allow_download,
            "normalize_features": True,
            "preprocess": {"resize": 224, "center_crop": 224, "normalize": True},
        }
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path, output_dir


def make_image(path: Path) -> None:
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path, format="PNG")


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
