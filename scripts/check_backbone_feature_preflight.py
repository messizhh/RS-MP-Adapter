#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_backbone_config_preflight import first_configured_path, normalize_weight_path
from scripts.check_backbone_model_load_preflight import (
    checkpoint_report_fields,
    collect_runtime_metadata,
    default_load_metadata,
    metadata_from_exception,
    metadata_from_model,
)
from src.config.config_loader import load_yaml_config
from src.features.image_preprocess import load_rgb_image
from src.models.base_backbone import BackboneUnavailableError, create_backbone
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny image feature preflight for one explicit image path.")
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--image-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_backbone_feature_preflight(
        backbone_config_path=args.backbone_config,
        expected_backbone=args.backbone,
        weights_path_override=args.weights_path,
        image_paths=args.image_path,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        source_script="scripts/check_backbone_feature_preflight.py",
    )
    print(f"backbone_feature_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_backbone_feature_preflight(
    *,
    backbone_config_path: str | Path,
    expected_backbone: str,
    weights_path_override: str,
    image_paths: list[str | Path],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    device: str,
    source_script: str = "scripts/check_backbone_feature_preflight.py",
) -> tuple[Path, bool]:
    config_path = Path(backbone_config_path)
    config = load_yaml_config(config_path)
    effective_config = copy.deepcopy(config)
    backbone = effective_config.get("backbone", {})
    errors: list[str] = []
    warnings: list[str] = []
    runtime_metadata = collect_runtime_metadata(device)
    load_metadata = default_load_metadata()
    loads_model = False
    reads_image_pixels = False
    extracts_features = False
    feature_shape: list[int] = []
    feature_dtype = None
    feature_norm = None
    feature_is_finite = False
    model_load_time_sec = 0.0
    preprocess_time_sec = 0.0
    encode_time_sec = 0.0

    if not isinstance(backbone, dict):
        backbone = {}
        effective_config["backbone"] = backbone
        errors.append("backbone config root must contain a backbone mapping")

    name = backbone.get("name")
    family = backbone.get("family")
    allow_download = backbone.get("allow_download")
    original_weights_value = first_configured_path(backbone)
    original_weights_path = normalize_weight_path(config_path, original_weights_value)
    resolved_weights_path = normalize_weight_path(Path.cwd() / "cli_override.yaml", weights_path_override)
    weights_exists = resolved_weights_path.exists() if resolved_weights_path is not None else False
    normalized_image_paths = [Path(path) for path in image_paths]

    if name != expected_backbone:
        errors.append(f"backbone name mismatch: expected {expected_backbone}, found {name}")
    if family != "remoteclip":
        errors.append("feature preflight currently supports only RemoteCLIP")
    if allow_download is not False:
        errors.append("backbone.allow_download must be false")
    if len(normalized_image_paths) != 1:
        errors.append("feature preflight requires exactly one explicit --image-path")
    if resolved_weights_path is None or not weights_exists:
        errors.append(f"configured weights path does not exist: {resolved_weights_path}")
    for image_path in normalized_image_paths:
        if not image_path.exists():
            errors.append(f"image path does not exist: {image_path}")

    model = None
    if not errors:
        backbone["weights"] = str(resolved_weights_path)
        backbone.pop("pretrained_path", None)
        try:
            start = time.perf_counter()
            model = create_backbone(expected_backbone, effective_config, dry_run=False, device=device)
            model.load_model().eval()
            model_load_time_sec = time.perf_counter() - start
            loads_model = True
            load_metadata = metadata_from_model(model, dry_run=False)
        except BackboneUnavailableError as exc:
            errors.append(str(exc))
            load_metadata = metadata_from_exception(exc)
        except Exception as exc:
            errors.append(f"model load failed: {exc}")

    if not errors and not load_metadata["checkpoint_loaded"]:
        errors.append("model object was created but no local checkpoint was confirmed as loaded")

    if not errors and model is not None:
        try:
            image_size = image_size_from_backbone(backbone)
            start = time.perf_counter()
            image = load_rgb_image(normalized_image_paths[0], image_size=image_size)
            reads_image_pixels = True
            image_tensor = pil_to_clip_tensor(image, device=device)
            preprocess_time_sec = time.perf_counter() - start
            start = time.perf_counter()
            feature = model.encode_image_preflight(image_tensor)
            encode_time_sec = time.perf_counter() - start
            extracts_features = True
            feature_shape = list(feature.shape)
            feature_dtype = str(feature.dtype)
            feature_norm = float(feature.norm().detach().cpu().item())
            feature_is_finite = bool(feature.isfinite().all().detach().cpu().item())
            if not feature_is_finite:
                errors.append("image feature contains non-finite values")
            if feature_norm is None or not math.isfinite(feature_norm):
                errors.append("image feature norm is not finite")
        except Exception as exc:
            errors.append(f"image feature preflight failed: {exc}")

    is_valid = not errors
    load_report_fields = checkpoint_report_fields(load_metadata, "cli_override")
    report = {
        "backbone_config_path": str(config_path),
        "backbone": name,
        "family": family,
        "feature_dim": backbone.get("feature_dim"),
        "image_size": backbone.get("image_size"),
        "weights_path": str(resolved_weights_path) if resolved_weights_path is not None else None,
        "weights_source": "cli_override",
        "original_weights_path": str(original_weights_path) if original_weights_path is not None else None,
        "resolved_weights_path": str(resolved_weights_path) if resolved_weights_path is not None else None,
        "weights_exists": weights_exists,
        "allow_download": allow_download,
        "normalize_features": backbone.get("normalize_features"),
        "preprocess": backbone.get("preprocess") if isinstance(backbone.get("preprocess"), dict) else {},
        "device": device,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "image_paths": [str(path) for path in normalized_image_paths],
        "image_count": len(normalized_image_paths),
        "feature_shape": feature_shape,
        "feature_dtype": feature_dtype,
        "feature_norm": feature_norm,
        "feature_is_finite": feature_is_finite,
        "model_load_time_sec": model_load_time_sec,
        "preprocess_time_sec": preprocess_time_sec,
        "encode_time_sec": encode_time_sec,
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
        "reads_image_pixels": reads_image_pixels,
        "extracts_features": extracts_features,
        "extracts_text_features": False,
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
    run_dir = unique_dir(Path(output_dir) / str(name or expected_backbone) / "backbone_feature_preflight")
    report_path = safe_write_json(run_dir / "backbone_feature_preflight_report.json", report, overwrite=False)
    return report_path, is_valid


def image_size_from_backbone(backbone: dict[str, Any]) -> int:
    preprocess = backbone.get("preprocess", {})
    if isinstance(preprocess, dict) and isinstance(preprocess.get("resize"), int):
        return preprocess["resize"]
    if isinstance(backbone.get("image_size"), int):
        return backbone["image_size"]
    return 224


def pil_to_clip_tensor(image: Any, device: str) -> Any:
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique backbone feature preflight directory under {base}")


if __name__ == "__main__":
    main()
