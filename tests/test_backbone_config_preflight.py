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
            self.assertEqual(report["weights_source"], "none")
            self.assertFalse(report["override_used"])
            self.assertIn("not ready", report["warnings"][0])

    def test_require_weights_with_null_weights_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path, output_dir = make_inputs(Path(temp_dir), weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir, require_weights=True, check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertIn("required", report["errors"][0])

    def test_cli_weights_path_existing_file_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "override.pt"
            weights_path.write_bytes(b"placeholder weights\n")
            config_path, output_dir = make_inputs(base, weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir, weights_path=weights_path, require_weights=True)

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["weights_source"], "cli_override")
            self.assertTrue(report["override_used"])
            self.assertEqual(report["resolved_weights_path"], str(weights_path))
            self.assertTrue(report["is_ready_for_real_model_load"])
            self.assertFalse(report["loads_model"])

    def test_cli_weights_path_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path, output_dir = make_inputs(base, weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir, weights_path=base / "missing.pt", check=False)

            self.assertNotEqual(completed.returncode, 0)
            report = read_json(extract_path(completed.stdout))
            self.assertFalse(report["is_valid"])
            self.assertEqual(report["weights_source"], "cli_override")
            self.assertIn("does not exist", report["errors"][0])

    def test_env_config_backbone_weight_path_existing_file_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "env_weights.pt"
            weights_path.write_bytes(b"placeholder weights\n")
            env_config_path = base / "env.yaml"
            write_env_config(env_config_path, "fake_backbone", str(weights_path))
            config_path, output_dir = make_inputs(base, weights=None, allow_download=False)

            completed = run_preflight(config_path, output_dir, env_config_path=env_config_path, require_weights=True)

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["weights_source"], "env_config")
            self.assertTrue(report["override_used"])
            self.assertEqual(report["env_config_path"], str(env_config_path))
            self.assertTrue(report["is_ready_for_real_model_load"])

    def test_cli_weights_path_has_priority_over_env_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            cli_weights_path = base / "cli.pt"
            env_weights_path = base / "env.pt"
            cli_weights_path.write_bytes(b"cli placeholder\n")
            env_weights_path.write_bytes(b"env placeholder\n")
            env_config_path = base / "env.yaml"
            write_env_config(env_config_path, "fake_backbone", str(env_weights_path))
            config_path, output_dir = make_inputs(base, weights=None, allow_download=False)

            completed = run_preflight(
                config_path,
                output_dir,
                weights_path=cli_weights_path,
                env_config_path=env_config_path,
                require_weights=True,
            )

            report = read_json(extract_path(completed.stdout))
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["weights_source"], "cli_override")
            self.assertEqual(report["resolved_weights_path"], str(cli_weights_path))

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
            self.assertEqual(report["weights_source"], "backbone_config")
            self.assertFalse(report["override_used"])
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

    def test_overrides_do_not_rewrite_backbone_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            weights_path = base / "override.pt"
            weights_path.write_bytes(b"placeholder weights\n")
            config_path, output_dir = make_inputs(base, weights=None, allow_download=False)
            before = config_path.read_text(encoding="utf-8")

            run_preflight(config_path, output_dir, weights_path=weights_path, require_weights=True)

            self.assertEqual(config_path.read_text(encoding="utf-8"), before)

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


def write_env_config(path: Path, backbone: str, weights_path: str) -> None:
    data = {"paths": {"backbone_weights": {backbone: weights_path}}}
    path.write_text(json.dumps(data), encoding="utf-8")


def run_preflight(
    config_path: Path,
    output_dir: Path,
    require_weights: bool = False,
    weights_path: Path | None = None,
    env_config_path: Path | None = None,
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
    if weights_path is not None:
        command.extend(["--weights-path", str(weights_path)])
    if env_config_path is not None:
        command.extend(["--env-config", str(env_config_path)])
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
