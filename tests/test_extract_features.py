from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from src.features.extract_features import run_guarded_real_feature_extraction
from src.features.feature_cache import load_feature_cache
from src.features.extract_features import run_dry_run_feature_extraction
from src.utils.io import read_json


class ExtractFeaturesTest(unittest.TestCase):
    def test_dry_run_extraction_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_dry_run_feature_extraction(
                dataset="eurosat",
                backbone_name="fake_backbone",
                backbone_config={"backbone": {"name": "fake_backbone", "family": "fake", "feature_dim": 8}},
                output_dir=temp_dir,
                split_path=None,
                max_samples=12,
                batch_size=4,
                device="cpu",
                execution_env="local_wsl",
                run_mode="smoke_test",
                prompt_templates=["a satellite photo of {}.", "a remote sensing image of {}."],
            )
            cache = load_feature_cache(result["cache_path"])
            self.assertEqual(len(cache.image_features), 12)
            self.assertEqual(len(cache.image_features[0]), 8)
            self.assertEqual(len(cache.text_features), 3)
            self.assertEqual(len(cache.text_prompts), 6)
            self.assertFalse(cache.metadata["is_paper_result"])
            summary = read_json(result["summary_path"])
            self.assertFalse(summary["is_paper_result"])

    def test_extract_features_cli_and_validation_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "features"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/extract_features.py",
                    "--dataset",
                    "eurosat",
                    "--backbone",
                    "fake_backbone",
                    "--dry-run",
                    "--max-samples",
                    "12",
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cache_path = extract_path(completed.stdout, "feature_cache_path")
            validation = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_feature_cache.py",
                    "--feature-cache",
                    str(cache_path),
                    "--output-dir",
                    str(Path(temp_dir) / "validation"),
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "smoke_test",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report_path = extract_path(validation.stdout, "validation_report_path")
            report = read_json(report_path)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["num_images"], 12)
            self.assertEqual(report["feature_dim"], 8)
            self.assertTrue(report["uses_fake_features"])

    def test_extraction_unique_paths_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            kwargs = dict(
                dataset="eurosat",
                backbone_name="fake_backbone",
                backbone_config={"backbone": {"name": "fake_backbone", "family": "fake", "feature_dim": 8}},
                output_dir=temp_dir,
                split_path=None,
                max_samples=3,
                batch_size=2,
                device="cpu",
                execution_env="local_wsl",
                run_mode="smoke_test",
            )
            first = run_dry_run_feature_extraction(**kwargs)
            second = run_dry_run_feature_extraction(**kwargs)
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertTrue(first["cache_path"].exists())
            self.assertTrue(second["cache_path"].exists())

    def test_non_dry_run_requires_explicit_real_extraction_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/extract_features.py",
                    "--dataset",
                    "eurosat",
                    "--backbone",
                    "remoteclip_vit_b32",
                    "--output-dir",
                    temp_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("--allow-real-extraction", completed.stderr)

    def test_guarded_real_extraction_writes_server_metadata_without_text_or_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset_root = base / "dataset"
            image_a = dataset_root / "class_a" / "a.png"
            image_b = dataset_root / "class_b" / "b.png"
            make_image(image_a)
            make_image(image_b)
            split_path = base / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "class_to_idx": {"class_a": 0, "class_b": 1},
                        "test": [
                            {"path": "class_a/a.png", "label": 0},
                            {"path": "class_b/b.png", "label": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            weights_path = base / "remoteclip.pt"
            weights_path.write_bytes(b"placeholder\n")
            backbone_config = {
                "backbone": {
                    "name": "remoteclip_vit_b32",
                    "family": "remoteclip",
                    "feature_dim": 512,
                    "image_size": 224,
                    "weights": None,
                    "allow_download": False,
                    "normalize_features": True,
                }
            }

            with patched_remoteclip_modules(checkpoint={"state_dict": {"visual.proj": torch.ones(1)}}):
                result = run_guarded_real_feature_extraction(
                    dataset="eurosat",
                    backbone_name="remoteclip_vit_b32",
                    backbone_config=backbone_config,
                    output_dir=base / "outputs" / "features",
                    split_path=split_path,
                    split_section="test",
                    dataset_root=dataset_root,
                    weights_path=weights_path,
                    batch_size=1,
                    device="cpu",
                    execution_env="remote_server",
                    run_mode="server_full",
                    command="pytest guarded real extraction",
                    is_paper_result_candidate=True,
                )

            cache = load_feature_cache(result["cache_path"])
            summary = read_json(result["summary_path"])
            self.assertEqual(tuple(cache.image_features.shape), (2, 512))
            self.assertIsNone(cache.text_features)
            self.assertIsNone(cache.text_prompts)
            self.assertEqual(cache.image_labels.tolist(), [0, 1])
            self.assertIn("/outputs/features/", str(result["cache_path"]))
            self.assertFalse(cache.metadata["is_paper_result"])
            self.assertTrue(cache.metadata["is_paper_result_candidate"])
            self.assertFalse(cache.metadata["eligible_for_paper_tables"])
            self.assertEqual(cache.metadata["execution_env"], "remote_server")
            self.assertEqual(cache.metadata["run_mode"], "server_full")
            self.assertEqual(cache.metadata["split_path"], str(split_path))
            self.assertEqual(cache.metadata["split_section"], "test")
            self.assertEqual(cache.metadata["image_count"], 2)
            self.assertEqual(cache.metadata["feature_shape"], [2, 512])
            self.assertEqual(cache.metadata["weights_source"], "cli_override")
            self.assertTrue(cache.metadata["checkpoint_loaded"])
            self.assertIn("start_time", cache.metadata)
            self.assertIn("end_time", cache.metadata)
            self.assertFalse(cache.metadata["extracts_text_features"])
            self.assertFalse(cache.metadata["saves_predictions"])
            self.assertFalse(cache.metadata["trains_model"])
            self.assertFalse(cache.metadata["evaluates_model"])
            self.assertEqual(summary["feature_shape"], [2, 512])

    def test_guarded_real_extraction_rejects_local_or_tiny_modes_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dataset_root = base / "dataset"
            dataset_root.mkdir()
            split_path = base / "split.json"
            split_path.write_text(json.dumps({"class_to_idx": {"class_a": 0}, "test": []}), encoding="utf-8")
            weights_path = base / "remoteclip.pt"
            weights_path.write_bytes(b"placeholder\n")
            with self.assertRaisesRegex(ValueError, "remote_server"):
                run_guarded_real_feature_extraction(
                    dataset="eurosat",
                    backbone_name="remoteclip_vit_b32",
                    backbone_config={"backbone": {"name": "remoteclip_vit_b32", "family": "remoteclip"}},
                    output_dir=base / "outputs" / "features",
                    split_path=split_path,
                    split_section="test",
                    dataset_root=dataset_root,
                    weights_path=weights_path,
                    batch_size=1,
                    device="cpu",
                    execution_env="local_wsl",
                    run_mode="tiny_subset",
                )


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in output: {stdout}")


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(10, 30, 20)).save(path, format="PNG")


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
