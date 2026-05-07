#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely verify backbone config and weight path readiness.")
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    parser.add_argument("--require-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_backbone_config_preflight(
        backbone_config_path=args.backbone_config,
        expected_backbone=args.backbone,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        require_weights=args.require_weights,
        source_script="scripts/check_backbone_config_preflight.py",
    )
    print(f"backbone_config_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_backbone_config_preflight(
    *,
    backbone_config_path: str | Path,
    expected_backbone: str,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    require_weights: bool,
    source_script: str,
) -> tuple[Path, bool]:
    config_path = Path(backbone_config_path)
    config = load_yaml_config(config_path)
    backbone = config.get("backbone", {})
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(backbone, dict):
        backbone = {}
        errors.append("backbone config root must contain a backbone mapping")

    name = backbone.get("name")
    family = backbone.get("family")
    feature_dim = backbone.get("feature_dim")
    image_size = backbone.get("image_size")
    allow_download = backbone.get("allow_download")
    normalize_features = backbone.get("normalize_features")
    preprocess = backbone.get("preprocess", {})
    weights_value = first_configured_path(backbone)
    weights_path = normalize_weight_path(config_path, weights_value)
    weights_configured = weights_path is not None
    weights_exists = weights_path.exists() if weights_path is not None else False

    if name != expected_backbone:
        errors.append(f"backbone name mismatch: expected {expected_backbone}, found {name}")
    if allow_download is not False:
        errors.append("backbone.allow_download must be false")
    if require_weights and not weights_configured:
        errors.append("weights or pretrained_path is required but not configured")
    if weights_configured and not weights_exists:
        errors.append(f"configured weights path does not exist: {weights_path}")
    if not weights_configured:
        warnings.append("weights are not configured; config is not ready for real model loading")

    is_ready_for_real_model_load = bool(weights_configured and weights_exists and allow_download is False)
    is_valid = not errors
    report = {
        "backbone_config_path": str(config_path),
        "backbone": name,
        "family": family,
        "feature_dim": feature_dim,
        "image_size": image_size,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "weights_configured": weights_configured,
        "weights_exists": weights_exists,
        "allow_download": allow_download,
        "normalize_features": normalize_features,
        "preprocess": preprocess if isinstance(preprocess, dict) else {},
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "downloads_weights": False,
        "is_ready_for_real_model_load": is_ready_for_real_model_load,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "source_script": source_script,
        "created_at": utc_now_iso(),
    }
    run_dir = unique_dir(Path(output_dir) / str(name or expected_backbone) / "backbone_config_preflight")
    report_path = safe_write_json(run_dir / "backbone_config_preflight_report.json", report, overwrite=False)
    return report_path, is_valid


def first_configured_path(backbone: dict[str, Any]) -> str | None:
    for key in ("weights", "pretrained_path"):
        value = backbone.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def normalize_weight_path(config_path: Path, weights_value: str | None) -> Path | None:
    if weights_value is None:
        return None
    path = Path(weights_value)
    if path.is_absolute():
        return path
    return config_path.parent / path


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique backbone config preflight directory under {base}")


if __name__ == "__main__":
    main()
