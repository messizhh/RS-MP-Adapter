from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.io import read_json


class BackboneConfigPreflightTest(unittest.TestCase):
    def test_null_weights_valid_without_require_but_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir)

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertFalse(report["weights_configured"])
            self.assertFalse(report["weights_exists"])
            self.assertFalse(report["is_ready_for_real_model_load"])
            self.assertIn("not ready", report["warnings"][0])

    def test_require_weights_with_null_weights_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir, require_weights=True, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertIn("required", report["errors"][0])

    def test_missing_weights_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights="missing.pt", allow_download=False)

            completed = run_preflight(config_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertTrue(report["weights_configured"])
            self.assertFalse(report["weights_exists"])
            self.assertIn("does not exist", report["errors"][0])

    def test_existing_weights_path_valid_ready_but_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "weights.pt").write_bytes(b"placeholder weights\n")
            config_path, output_dir = make_inputs(base, weights="weights.pt", allow_download=False)

            completed = run_preflight(config_path, output_dir, require_weights=True)

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["weights_configured"])
            self.assertTrue(report["weights_exists"])
            self.assertTrue(report["is_ready_for_real_model_load"])
            self.assertFalse(report["loads_model"])

    def test_allow_download_true_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=True)

            completed = run_preflight(config_path, output_dir, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertIn("allow_download", report["errors"][0])

    def test_report_safety_fields_are_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir)
            report = read_json(extract_path(completed.stdout))

            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["downloads_weights"])

    def test_output_path_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=False)

            first = run_preflight(config_path, output_dir)
            second = run_preflight(config_path, output_dir)

            first_path = extract_path(first.stdout)
            second_path = extract_path(second.stdout)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())


def make_inputs(base: Path, weights: str | None, allow_download: bool) -> tuple[Path, Path]:
    config_path = base / "backbone.yaml"
    output_dir = base / "reports"
    write_config(config_path, weights=weights, allow_download=allow_download)
    return config_path, output_dir


def write_config(path: Path, weights: str | None, allow_download: bool) -> None:
    data = {
        "backbone": {
            "name": "fake_backbone",
            "family": "fake",
            "feature_dim": 8,
            "image_size": 224,
            "weights": weights,
            "allow_download": allow_download,
            "normalize_features": True,
            "preprocess": {"resize": 224, "center_crop": 224, "normalize": True},
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def run_preflight(
    config_path: Path,
    output_dir: Path,
    require_weights: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/check_backbone_config_preflight.py",
        "--backbone-config",
        str(config_path),
        "--backbone",
        "fake_backbone",
        "--output-dir",
        str(output_dir),
        "--execution-env",
        "local_wsl",
        "--run-mode",
        "local_validation",
    ]
    if require_weights:
        command.append("--require-weights")
    return subprocess.run(command, check=check, capture_output=True, text=True)


def extract_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("backbone_config_report_path="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing backbone_config_report_path in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
