from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_json


class BackboneModelLoadPreflightTest(unittest.TestCase):
    def test_missing_weights_fails_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(
                Path(temp_dir),
                name="remoteclip_vit_b32",
                family="remoteclip",
                weights=None,
                allow_download=False,
            )

            completed = run_preflight(config_path, output_dir, backbone="remoteclip_vit_b32", check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertFalse(report["loads_model"])
            self.assertEqual(report["weights_source"], "none")
            self.assertIn("requires a resolved local weights path", report["errors"][0])

    def test_missing_cli_weights_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path, output_dir = make_inputs(
                base,
                name="remoteclip_vit_b32",
                family="remoteclip",
                weights=None,
                allow_download=False,
            )

            completed = run_preflight(
                config_path,
                output_dir,
                backbone="remoteclip_vit_b32",
                weights_path=base / "missing.pt",
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertFalse(report["loads_model"])
            self.assertEqual(report["weights_source"], "cli_override")
            self.assertIn("does not exist", report["errors"][0])

    def test_allow_download_true_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(
                Path(temp_dir),
                name="remoteclip_vit_b32",
                family="remoteclip",
                weights=None,
                allow_download=True,
            )

            completed = run_preflight(config_path, output_dir, backbone="remoteclip_vit_b32", check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertFalse(report["loads_model"])
            self.assertIn("allow_download", report["errors"][0])

    def test_fake_dry_run_load_preflight_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(
                Path(temp_dir),
                name="fake_backbone",
                family="fake",
                weights=None,
                allow_download=False,
            )

            completed = run_preflight(config_path, output_dir, backbone="fake_backbone", dry_run=True)

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["reads_image_pixels"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["is_paper_result"])
            self.assertEqual(report["execution_env"], "local_wsl")
            self.assertEqual(report["run_mode"], "local_validation")
            self.assertEqual(report["weights_source"], "none")

    def test_report_output_path_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(
                Path(temp_dir),
                name="fake_backbone",
                family="fake",
                weights=None,
                allow_download=False,
            )

            first = run_preflight(config_path, output_dir, backbone="fake_backbone", dry_run=True)
            second = run_preflight(config_path, output_dir, backbone="fake_backbone", dry_run=True)

            first_path = extract_path(first.stdout)
            second_path = extract_path(second.stdout)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())


def make_inputs(
    base: Path,
    *,
    name: str,
    family: str,
    weights: str | None,
    allow_download: bool,
) -> tuple[Path, Path]:
    config_path = base / "backbone.yaml"
    output_dir = base / "reports"
    data = {
        "backbone": {
            "name": name,
            "family": family,
            "feature_dim": 8 if family == "fake" else 512,
            "image_size": 224,
            "weights": weights,
            "allow_download": allow_download,
            "normalize_features": True,
            "preprocess": {"resize": 224, "center_crop": 224, "normalize": True},
        }
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path, output_dir


def run_preflight(
    config_path: Path,
    output_dir: Path,
    *,
    backbone: str,
    weights_path: Path | None = None,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/check_backbone_model_load_preflight.py",
        "--backbone-config",
        str(config_path),
        "--backbone",
        backbone,
        "--output-dir",
        str(output_dir),
        "--execution-env",
        "local_wsl",
        "--run-mode",
        "local_validation",
        "--device",
        "cpu",
    ]
    if weights_path is not None:
        command.extend(["--weights-path", str(weights_path)])
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(command, check=check, capture_output=True, text=True)


def extract_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("backbone_model_load_report_path="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing backbone_model_load_report_path in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
