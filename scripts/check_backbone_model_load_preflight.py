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
    load_metadata = default_load_metadata()
    runtime_metadata = collect_runtime_metadata(device)

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
            load_metadata = metadata_from_model(model, dry_run=dry_run)
        except BackboneUnavailableError as exc:
            errors.append(str(exc))
            load_metadata = metadata_from_exception(exc)
        except Exception as exc:
            errors.append(f"model load failed: {exc}")

    if not errors and not dry_run and family != "fake" and not load_metadata["checkpoint_loaded"]:
        errors.append("model object was created but no local checkpoint was confirmed as loaded")

    is_valid = not errors
    load_report_fields = checkpoint_report_fields(load_metadata, weights_source)
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
        "checkpoint_loaded": load_metadata["checkpoint_loaded"],
        "checkpoint_num_tensors": load_metadata["checkpoint_num_tensors"],
        "checkpoint_load_mode": load_metadata["checkpoint_load_mode"],
        "missing_keys_count": load_metadata["missing_keys_count"],
        "unexpected_keys_count": load_metadata["unexpected_keys_count"],
        "missing_keys_sample": load_metadata["missing_keys_sample"],
        "unexpected_keys_sample": load_metadata["unexpected_keys_sample"],
        "model_class": load_metadata["model_class"],
        **load_report_fields,
        "torch_version": runtime_metadata["torch_version"],
        "open_clip_version": runtime_metadata["open_clip_version"],
        "cuda_available": runtime_metadata["cuda_available"],
        "cuda_device_name": runtime_metadata["cuda_device_name"],
        "extracts_features": False,
        "reads_image_pixels": False,
        "trains_model": False,
        "evaluates_model": False,
        "downloads_weights": False,
        "saves_feature_cache": False,
        "saves_predictions": False,
        "saves_logits": False,
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


def default_load_metadata() -> dict[str, Any]:
    return {
        "checkpoint_loaded": False,
        "checkpoint_num_tensors": 0,
        "checkpoint_load_mode": "not_attempted",
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
        "missing_keys_sample": [],
        "unexpected_keys_sample": [],
        "model_class": None,
        "open_clip_initial_pretrained": None,
        "open_clip_initialization_warning_expected": False,
        "checkpoint_load_happened_after_model_init": False,
        "final_weights_loaded_from_checkpoint": False,
        "final_weight_source": None,
        "final_checkpoint_load_status": "not_attempted",
    }


def metadata_from_model(model: Any, dry_run: bool) -> dict[str, Any]:
    metadata = default_load_metadata()
    model_metadata = getattr(model, "load_metadata", None)
    if isinstance(model_metadata, dict):
        metadata.update(model_metadata)
    elif dry_run:
        metadata["checkpoint_load_mode"] = "dry_run"
        metadata["model_class"] = model.__class__.__name__
    else:
        metadata["model_class"] = model.__class__.__name__
    return metadata


def checkpoint_report_fields(load_metadata: dict[str, Any], weights_source: str) -> dict[str, Any]:
    fields = {
        "open_clip_initial_pretrained": load_metadata.get("open_clip_initial_pretrained"),
        "open_clip_initialization_warning_expected": bool(
            load_metadata.get("open_clip_initialization_warning_expected", False)
        ),
        "checkpoint_load_happened_after_model_init": bool(
            load_metadata.get("checkpoint_load_happened_after_model_init", False)
        ),
        "final_weights_loaded_from_checkpoint": bool(load_metadata.get("final_weights_loaded_from_checkpoint", False)),
        "final_weight_source": load_metadata.get("final_weight_source"),
        "final_checkpoint_load_status": load_metadata.get("final_checkpoint_load_status", "not_attempted"),
    }
    if weights_source != "none" and fields["checkpoint_load_happened_after_model_init"]:
        fields["final_weight_source"] = f"{weights_source}_checkpoint"
    return fields


def metadata_from_exception(exc: BaseException) -> dict[str, Any]:
    metadata = default_load_metadata()
    exc_metadata = getattr(exc, "load_metadata", None)
    if isinstance(exc_metadata, dict):
        metadata.update(exc_metadata)
    return metadata


def collect_runtime_metadata(device: str) -> dict[str, Any]:
    torch_version = None
    open_clip_version = None
    cuda_available = False
    cuda_device_name = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and device.startswith("cuda"):
            cuda_device_name = torch.cuda.get_device_name(device)
    except Exception:
        pass
    try:
        import open_clip

        open_clip_version = getattr(open_clip, "__version__", "unknown")
    except Exception:
        pass
    return {
        "torch_version": torch_version,
        "open_clip_version": open_clip_version,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
    }


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
