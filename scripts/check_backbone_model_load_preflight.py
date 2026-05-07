#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_backbone_config_preflight import first_configured_path, normalize_weight_path
from src.config.config_loader import load_yaml_config
from src.models.base_backbone import BackboneUnavailableError, create_backbone
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny model-load preflight for a configured backbone.")
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_backbone_model_load_preflight(
        backbone_config_path=args.backbone_config,
        expected_backbone=args.backbone,
        weights_path_override=args.weights_path,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        dry_run=args.dry_run,
        source_script="scripts/check_backbone_model_load_preflight.py",
    )
    print(f"backbone_model_load_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_backbone_model_load_preflight(
    *,
    backbone_config_path: str | Path,
    expected_backbone: str,
    weights_path_override: str | None,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    device: str,
    dry_run: bool = False,
    source_script: str = "scripts/check_backbone_model_load_preflight.py",
) -> tuple[Path, bool]:
    config_path = Path(backbone_config_path)
    config = load_yaml_config(config_path)
    effective_config = copy.deepcopy(config)
    backbone = effective_config.get("backbone", {})
    errors: list[str] = []
    warnings: list[str] = []
    loads_model = False

    if not isinstance(backbone, dict):
        backbone = {}
        effective_config["backbone"] = backbone
        errors.append("backbone config root must contain a backbone mapping")

    name = backbone.get("name")
    family = backbone.get("family")
    allow_download = backbone.get("allow_download")
    original_weights_value = first_configured_path(backbone)
    original_weights_path = normalize_weight_path(config_path, original_weights_value)
    resolved_weights_path, weights_source = resolve_model_load_weight_path(
        config_path=config_path,
        original_weights_value=original_weights_value,
        weights_path_override=weights_path_override,
    )
    weights_configured = resolved_weights_path is not None
    weights_exists = resolved_weights_path.exists() if resolved_weights_path is not None else False

    if name != expected_backbone:
        errors.append(f"backbone name mismatch: expected {expected_backbone}, found {name}")
    if allow_download is not False:
        errors.append("backbone.allow_download must be false")
    if not dry_run and family != "fake" and not weights_configured:
        errors.append("model-load preflight requires a resolved local weights path")
    if weights_configured and not weights_exists:
        errors.append(f"configured weights path does not exist: {resolved_weights_path}")

    if not errors:
        if resolved_weights_path is not None:
            backbone["weights"] = str(resolved_weights_path)
            backbone.pop("pretrained_path", None)
        try:
            model = create_backbone(expected_backbone, effective_config, dry_run=dry_run, device=device)
            model.load_model().eval()
            loads_model = True
        except BackboneUnavailableError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"model load failed: {exc}")

    is_valid = not errors
    report = {
        "backbone_config_path": str(config_path),
        "backbone": name,
        "family": family,
        "feature_dim": backbone.get("feature_dim"),
        "image_size": backbone.get("image_size"),
        "weights_path": str(resolved_weights_path) if resolved_weights_path is not None else None,
        "weights_source": weights_source,
        "original_weights_path": str(original_weights_path) if original_weights_path is not None else None,
        "resolved_weights_path": str(resolved_weights_path) if resolved_weights_path is not None else None,
        "weights_configured": weights_configured,
        "weights_exists": weights_exists,
        "allow_download": allow_download,
        "normalize_features": backbone.get("normalize_features"),
        "preprocess": backbone.get("preprocess") if isinstance(backbone.get("preprocess"), dict) else {},
        "device": device,
        "dry_run": dry_run,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "loads_model": loads_model,
        "extracts_features": False,
        "reads_image_pixels": False,
        "trains_model": False,
        "evaluates_model": False,
        "downloads_weights": False,
        "saves_feature_cache": False,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "source_script": source_script,
        "created_at": utc_now_iso(),
    }
    run_dir = unique_dir(Path(output_dir) / str(name or expected_backbone) / "backbone_model_load_preflight")
    report_path = safe_write_json(run_dir / "backbone_model_load_preflight_report.json", report, overwrite=False)
    return report_path, is_valid


def resolve_model_load_weight_path(
    *,
    config_path: Path,
    original_weights_value: str | None,
    weights_path_override: str | None,
) -> tuple[Path | None, str]:
    if weights_path_override:
        return normalize_weight_path(Path.cwd() / "cli_override.yaml", weights_path_override), "cli_override"
    if original_weights_value:
        return normalize_weight_path(config_path, original_weights_value), "backbone_config"
    return None, "none"


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique backbone model-load preflight directory under {base}")


if __name__ == "__main__":
    main()
