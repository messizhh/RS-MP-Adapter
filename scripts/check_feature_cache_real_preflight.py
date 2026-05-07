#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_backbone_config_preflight import first_configured_path, normalize_weight_path
from scripts.check_backbone_feature_preflight import image_size_from_backbone, pil_to_clip_tensor
from scripts.check_backbone_model_load_preflight import (
    checkpoint_report_fields,
    collect_runtime_metadata,
    default_load_metadata,
    metadata_from_exception,
    metadata_from_model,
)
from src.config.config_loader import load_yaml_config
from src.features.feature_cache import FeatureCache, save_feature_cache
from src.features.image_preprocess import load_rgb_image
from src.models.base_backbone import BackboneUnavailableError, create_backbone
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


MAX_ALLOWED_IMAGES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny real image feature-cache preflight for explicit image paths.")
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--image-path", action="append", default=[])
    parser.add_argument("--image-paths-file", default=None)
    parser.add_argument("--max-images", type=int, default=MAX_ALLOWED_IMAGES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_feature_cache_real_preflight(
        backbone_config_path=args.backbone_config,
        expected_backbone=args.backbone,
        weights_path_override=args.weights_path,
        image_paths=args.image_path,
        image_paths_file=args.image_paths_file,
        max_images=args.max_images,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        source_script="scripts/check_feature_cache_real_preflight.py",
    )
    print(f"feature_cache_real_preflight_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_feature_cache_real_preflight(
    *,
    backbone_config_path: str | Path,
    expected_backbone: str,
    weights_path_override: str,
    image_paths: list[str | Path],
    image_paths_file: str | Path | None,
    max_images: int,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    device: str,
    source_script: str = "scripts/check_feature_cache_real_preflight.py",
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
    saves_feature_cache = False
    feature_cache_path: Path | None = None
    feature_shape: list[int] = []
    feature_dtype = None
    feature_norm_stats = {"min": None, "max": None, "mean": None}
    feature_is_finite = False
    model_load_time_sec = 0.0
    preprocess_time_sec = 0.0
    encode_time_sec = 0.0
    save_time_sec = 0.0

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
    output_root = Path(output_dir)
    normalized_image_paths = collect_image_paths(image_paths, image_paths_file)

    if name != expected_backbone:
        errors.append(f"backbone name mismatch: expected {expected_backbone}, found {name}")
    if family != "remoteclip":
        errors.append("tiny feature-cache preflight currently supports only RemoteCLIP")
    if allow_download is not False:
        errors.append("backbone.allow_download must be false")
    if max_images <= 0 or max_images > MAX_ALLOWED_IMAGES:
        errors.append(f"--max-images must be between 1 and {MAX_ALLOWED_IMAGES}")
    if not normalized_image_paths:
        errors.append("at least one explicit image path is required")
    if len(normalized_image_paths) > max_images:
        errors.append(f"received {len(normalized_image_paths)} image paths, exceeding --max-images={max_images}")
    if len(normalized_image_paths) > MAX_ALLOWED_IMAGES:
        errors.append(f"received {len(normalized_image_paths)} image paths, exceeding hard cap {MAX_ALLOWED_IMAGES}")
    if output_targets_formal_features_dir(output_root):
        errors.append("tiny preflight feature caches must not be written under outputs/features")
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

    features = None
    if not errors and model is not None:
        try:
            image_size = image_size_from_backbone(backbone)
            tensors = []
            start = time.perf_counter()
            for image_path in normalized_image_paths:
                image = load_rgb_image(image_path, image_size=image_size)
                tensors.append(pil_to_clip_tensor(image, device=device))
            reads_image_pixels = True
            image_batch = torch.cat(tensors, dim=0)
            preprocess_time_sec = time.perf_counter() - start
            start = time.perf_counter()
            features = model.encode_image_preflight(image_batch)
            encode_time_sec = time.perf_counter() - start
            extracts_features = True
            feature_shape = list(features.shape)
            feature_dtype = str(features.dtype)
            norms = features.norm(dim=1).detach().cpu()
            feature_norm_stats = {
                "min": float(norms.min().item()),
                "max": float(norms.max().item()),
                "mean": float(norms.mean().item()),
            }
            feature_is_finite = bool(features.isfinite().all().detach().cpu().item())
            if not feature_is_finite:
                errors.append("image features contain non-finite values")
            if not all(math.isfinite(value) for value in feature_norm_stats.values() if value is not None):
                errors.append("image feature norm stats are not finite")
        except Exception as exc:
            errors.append(f"tiny feature-cache preflight failed: {exc}")

    load_report_fields = checkpoint_report_fields(load_metadata, "cli_override")

    if not errors and features is not None:
        try:
            run_dir = unique_dir(output_root / str(name or expected_backbone) / "feature_cache_real_preflight")
            feature_cache_path = run_dir / "tiny_preflight_feature_cache.pt"
            start = time.perf_counter()
            cache = FeatureCache(
                image_features=features.detach().cpu(),
                image_labels=torch.zeros(len(normalized_image_paths), dtype=torch.long),
                image_paths=[str(path) for path in normalized_image_paths],
                split_name="tiny_real_preflight",
                class_to_idx={"preflight_class": 0},
                text_features=None,
                text_prompts=None,
                backbone=str(name),
                dataset="tiny_real_preflight",
                feature_dim=int(features.shape[1]),
                normalize_features=bool(backbone.get("normalize_features", True)),
                created_at=utc_now_iso(),
                source_script=source_script,
                metadata={
                    "feature_cache_is_tiny_preflight": True,
                    "is_real_feature_extraction": False,
                    "is_full_feature_extraction": False,
                    "is_paper_result": False,
                    "eligible_for_paper_tables": False,
                    "execution_env": execution_env,
                    "run_mode": run_mode,
                    "loads_model": loads_model,
                    "checkpoint_loaded": load_metadata["checkpoint_loaded"],
                    **load_report_fields,
                    "reads_image_pixels": reads_image_pixels,
                    "extracts_features": extracts_features,
                    "extracts_text_features": False,
                    "saves_predictions": False,
                    "saves_logits": False,
                    "trains_model": False,
                    "evaluates_model": False,
                    "downloads_weights": False,
                },
            )
            save_feature_cache(cache, feature_cache_path)
            save_time_sec = time.perf_counter() - start
            saves_feature_cache = True
        except Exception as exc:
            errors.append(f"failed to save tiny preflight feature cache: {exc}")

    is_valid = not errors
    if feature_cache_path is None:
        run_dir = unique_dir(output_root / str(name or expected_backbone) / "feature_cache_real_preflight")
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
        "max_images": max_images,
        "feature_shape": feature_shape,
        "feature_dtype": feature_dtype,
        "feature_norm_stats": feature_norm_stats,
        "feature_is_finite": feature_is_finite,
        "feature_cache_path": str(feature_cache_path) if feature_cache_path is not None else None,
        "feature_cache_is_tiny_preflight": True,
        "model_load_time_sec": model_load_time_sec,
        "preprocess_time_sec": preprocess_time_sec,
        "encode_time_sec": encode_time_sec,
        "save_time_sec": save_time_sec,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
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
        "saves_feature_cache": saves_feature_cache,
        "saves_predictions": False,
        "saves_logits": False,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "source_script": source_script,
        "created_at": utc_now_iso(),
    }
    report_path = safe_write_json(run_dir / "feature_cache_real_preflight_report.json", report, overwrite=False)
    return report_path, is_valid


def collect_image_paths(image_paths: list[str | Path], image_paths_file: str | Path | None) -> list[Path]:
    paths = [Path(path) for path in image_paths]
    if image_paths_file is not None:
        source = Path(image_paths_file)
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                paths.append(Path(stripped))
    return paths


def output_targets_formal_features_dir(output_dir: Path) -> bool:
    parts = output_dir.parts
    return any(parts[index : index + 2] == ("outputs", "features") for index in range(len(parts) - 1))


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique tiny feature-cache preflight directory under {base}")


if __name__ == "__main__":
    main()
