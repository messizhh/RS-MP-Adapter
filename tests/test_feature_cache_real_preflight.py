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

from scripts.check_feature_cache_real_preflight import run_feature_cache_real_preflight
from src.features.feature_cache import load_feature_cache
from src.utils.io import read_json


class FeatureCacheRealPreflightTest(unittest.TestCase):
    def test_tiny_feature_cache_preflight_writes_cache_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_a = base / "a.png"
            image_b = base / "b.png"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_a)
            make_image(image_b)
            config_path, output_dir = make_inputs(base)
            checkpoint = {"state_dict": {"visual.proj": torch.ones(1), "text_projection": torch.ones(1)}}

            with patched_remoteclip_modules(checkpoint=checkpoint):
                report_path, is_valid = run_feature_cache_real_preflight(
                    backbone_config_path=config_path,
                    expected_backbone="remoteclip_vit_b32",
                    weights_path_override=str(weights_path),
                    image_paths=[image_a, image_b],
                    image_paths_file=None,
                    max_images=4,
                    output_dir=output_dir,
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    device="cpu",
                )

            report = read_json(report_path)
            self.assertTrue(is_valid)
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
            self.assertTrue(report["saves_feature_cache"])
            self.assertTrue(report["feature_cache_is_tiny_preflight"])
            self.assertFalse(report["extracts_text_features"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["saves_logits"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["eligible_for_paper_tables"])
            self.assertEqual(report["image_count"], 2)
            self.assertEqual(report["max_images"], 4)
            self.assertEqual(report["feature_shape"], [2, 512])
            self.assertEqual(report["feature_dtype"], "torch.float32")
            self.assertTrue(report["feature_is_finite"])
            self.assertGreater(report["feature_norm_stats"]["mean"], 0.0)
            self.assertNotIn("outputs/features", report["feature_cache_path"])

            cache = load_feature_cache(report["feature_cache_path"])
            self.assertEqual(tuple(cache.image_features.shape), (2, 512))
            self.assertTrue(cache.metadata["feature_cache_is_tiny_preflight"])
            self.assertFalse(cache.metadata["is_paper_result"])
            self.assertFalse(cache.metadata["eligible_for_paper_tables"])
            self.assertFalse(cache.metadata["extracts_text_features"])
            self.assertFalse(cache.metadata["saves_predictions"])
            self.assertFalse(cache.metadata["saves_logits"])
            self.assertTrue(cache.metadata["final_weights_loaded_from_checkpoint"])
            self.assertEqual(cache.metadata["final_checkpoint_load_status"], "loaded_strictly_matching_keys")

    def test_image_paths_file_is_supported_without_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_path = base / "a.png"
            paths_file = base / "paths.txt"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_path)
            paths_file.write_text(f"# explicit paths only\n{image_path}\n", encoding="utf-8")
            config_path, output_dir = make_inputs(base)
            checkpoint = {"state_dict": {"visual.proj": torch.ones(1)}}

            with patched_remoteclip_modules(checkpoint=checkpoint):
                report_path, is_valid = run_feature_cache_real_preflight(
                    backbone_config_path=config_path,
                    expected_backbone="remoteclip_vit_b32",
                    weights_path_override=str(weights_path),
                    image_paths=[],
                    image_paths_file=paths_file,
                    max_images=1,
                    output_dir=output_dir,
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    device="cpu",
                )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["image_count"], 1)
            self.assertTrue(Path(report["feature_cache_path"]).exists())

    def test_max_images_rejects_too_many_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            weights_path.write_bytes(b"placeholder\n")
            image_paths = []
            for index in range(5):
                path = base / f"{index}.png"
                make_image(path)
                image_paths.append(path)
            config_path, output_dir = make_inputs(base)

            report_path, is_valid = run_feature_cache_real_preflight(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                image_paths=image_paths,
                image_paths_file=None,
                max_images=4,
                output_dir=output_dir,
                execution_env="local_wsl",
                run_mode="local_validation",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["saves_feature_cache"])
            self.assertIn("exceeding", " ".join(report["errors"]))

    def test_output_features_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "remoteclip.pt"
            image_path = base / "a.png"
            weights_path.write_bytes(b"placeholder\n")
            make_image(image_path)
            config_path, _ = make_inputs(base)

            report_path, is_valid = run_feature_cache_real_preflight(
                backbone_config_path=config_path,
                expected_backbone="remoteclip_vit_b32",
                weights_path_override=str(weights_path),
                image_paths=[image_path],
                image_paths_file=None,
                max_images=4,
                output_dir=base / "outputs" / "features",
                execution_env="local_wsl",
                run_mode="local_validation",
                device="cpu",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["saves_feature_cache"])
            self.assertIn("outputs/features", report["errors"][0])


def make_inputs(base: Path) -> tuple[Path, Path]:
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
