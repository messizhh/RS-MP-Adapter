#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_backbone_config_preflight import first_configured_path, normalize_weight_path
from scripts.check_backbone_feature_preflight import image_size_from_backbone, pil_to_clip_tensor
from scripts.check_backbone_model_load_preflight import (
    collect_runtime_metadata,
    default_load_metadata,
    metadata_from_exception,
    metadata_from_model,
)
from src.config.config_loader import load_yaml_config
from src.features.feature_cache import FeatureCache, save_feature_cache
from src.features.image_preprocess import load_rgb_image
from src.models.base_backbone import BackboneUnavailableError, create_backbone
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


ALLOWED_RUN_MODES = {"tiny_subset", "local_validation"}
MAX_ALLOWED_IMAGES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled tiny real image feature extraction runner.")
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-section", default="support", choices=["support", "train", "val", "test"])
    parser.add_argument("--image-list", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", required=True, choices=sorted(ALLOWED_RUN_MODES))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_tiny_real_feature_extraction(
        backbone_config_path=args.backbone_config,
        expected_backbone=args.backbone,
        weights_path_override=args.weights_path,
        dataset=args.dataset,
        split_path=args.split,
        split_section=args.split_section,
        image_list_path=args.image_list,
        dataset_root=args.dataset_root,
        max_images=args.max_images,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        command=shell_join(sys.argv),
        source_script="scripts/run_tiny_real_feature_extraction.py",
    )
    print(f"tiny_real_feature_extraction_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_tiny_real_feature_extraction(
    *,
    backbone_config_path: str | Path,
    expected_backbone: str,
    weights_path_override: str,
    dataset: str,
    split_path: str | Path | None,
    split_section: str,
    image_list_path: str | Path | None,
    dataset_root: str | Path | None,
    max_images: int,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    device: str,
    command: str | None = None,
    source_script: str = "scripts/run_tiny_real_feature_extraction.py",
) -> tuple[Path, bool]:
    config_path = Path(backbone_config_path)
    config = load_yaml_config(config_path)
    effective_config = copy.deepcopy(config)
    backbone = effective_config.get("backbone", {})
    output_root = Path(output_dir)
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
    resolved_weights_path = normalize_weight_path(Path.cwd() / "cli_override.yaml", weights_path_override)
    weights_exists = resolved_weights_path.exists() if resolved_weights_path is not None else False

    if name != expected_backbone:
        errors.append(f"backbone name mismatch: expected {expected_backbone}, found {name}")
    if family != "remoteclip":
        errors.append("tiny real feature extraction currently supports only RemoteCLIP")
    if allow_download is not False:
        errors.append("backbone.allow_download must be false")
    if run_mode not in ALLOWED_RUN_MODES:
        errors.append(f"--run-mode must be one of {sorted(ALLOWED_RUN_MODES)}")
    if max_images <= 0 or max_images > MAX_ALLOWED_IMAGES:
        errors.append(f"--max-images must be between 1 and {MAX_ALLOWED_IMAGES}")
    if not split_path and not image_list_path:
        errors.append("either --split or --image-list is required; dataset-root scanning is forbidden")
    if split_path and image_list_path:
        errors.append("use exactly one of --split or --image-list")
    if output_targets_formal_features_dir(output_root):
        errors.append("tiny real feature caches must not be written under outputs/features")
    if not output_targets_allowed_tiny_dir(output_root):
        errors.append("tiny real feature caches must be written under outputs/preflight or outputs/features_tiny_preflight")
    if resolved_weights_path is None or not weights_exists:
        errors.append(f"configured weights path does not exist: {resolved_weights_path}")

    selected = TinyImageSelection([], [], {"preflight_class": 0}, "tiny_real")
    if not errors:
        try:
            selected = collect_tiny_images(
                dataset=dataset,
                split_path=Path(split_path) if split_path else None,
                split_section=split_section,
                image_list_path=Path(image_list_path) if image_list_path else None,
                dataset_root=Path(dataset_root) if dataset_root else None,
                max_images=max_images,
            )
            if not selected.image_paths:
                errors.append("explicit input produced zero images")
            if len(selected.image_paths) > max_images:
                errors.append(f"selected {len(selected.image_paths)} images, exceeding --max-images={max_images}")
            for image_path in selected.image_paths:
                if not image_path.exists():
                    errors.append(f"image path does not exist: {image_path}")
        except Exception as exc:
            errors.append(f"failed to collect explicit tiny image inputs: {exc}")

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
            for image_path in selected.image_paths:
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
            errors.append(f"tiny real feature extraction failed: {exc}")

    metadata = {
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "dataset": dataset,
        "backbone": str(name or expected_backbone),
        "image_count": len(selected.image_paths),
        "max_images": max_images,
        "split_path": str(split_path) if split_path else None,
        "image_list_path": str(image_list_path) if image_list_path else None,
        "weights_source": "cli_override",
        "checkpoint_loaded": load_metadata["checkpoint_loaded"],
        "feature_shape": feature_shape,
        "feature_norm_stats": feature_norm_stats,
        "command": command or "",
        "git_commit": git_commit(),
        "feature_cache_is_tiny_preflight": True,
        "is_real_feature_extraction": True,
        "is_full_feature_extraction": False,
        "extracts_text_features": False,
        "saves_predictions": False,
        "trains_model": False,
        "evaluates_model": False,
        "downloads_weights": False,
    }

    if not errors and features is not None:
        try:
            run_dir = unique_dir(output_root / dataset / str(name or expected_backbone) / "tiny_real_feature_extraction")
            feature_cache_path = run_dir / "feature_cache.pt"
            start = time.perf_counter()
            cache = FeatureCache(
                image_features=features.detach().cpu(),
                image_labels=torch.tensor(selected.labels, dtype=torch.long),
                image_paths=[str(path) for path in selected.image_paths],
                split_name=selected.split_name,
                class_to_idx=selected.class_to_idx,
                text_features=None,
                text_prompts=None,
                backbone=str(name or expected_backbone),
                dataset=dataset,
                feature_dim=int(features.shape[1]),
                normalize_features=bool(backbone.get("normalize_features", True)),
                created_at=utc_now_iso(),
                source_script=source_script,
                metadata=metadata,
            )
            save_feature_cache(cache, feature_cache_path)
            save_time_sec = time.perf_counter() - start
            saves_feature_cache = True
        except Exception as exc:
            errors.append(f"failed to save tiny real feature cache: {exc}")

    if feature_cache_path is None:
        run_dir = unique_dir(output_root / dataset / str(name or expected_backbone) / "tiny_real_feature_extraction")
    is_valid = not errors
    report = {
        **metadata,
        "backbone_config_path": str(config_path),
        "family": family,
        "feature_dim": backbone.get("feature_dim"),
        "image_size": backbone.get("image_size"),
        "weights_exists": weights_exists,
        "allow_download": allow_download,
        "normalize_features": backbone.get("normalize_features"),
        "preprocess": backbone.get("preprocess") if isinstance(backbone.get("preprocess"), dict) else {},
        "device": device,
        "split_section": split_section,
        "image_paths": [str(path) for path in selected.image_paths],
        "feature_dtype": feature_dtype,
        "feature_is_finite": feature_is_finite,
        "feature_cache_path": str(feature_cache_path) if feature_cache_path is not None else None,
        "model_load_time_sec": model_load_time_sec,
        "preprocess_time_sec": preprocess_time_sec,
        "encode_time_sec": encode_time_sec,
        "save_time_sec": save_time_sec,
        "loads_model": loads_model,
        "reads_image_pixels": reads_image_pixels,
        "extracts_features": extracts_features,
        "saves_feature_cache": saves_feature_cache,
        "checkpoint_num_tensors": load_metadata["checkpoint_num_tensors"],
        "checkpoint_load_mode": load_metadata["checkpoint_load_mode"],
        "missing_keys_count": load_metadata["missing_keys_count"],
        "unexpected_keys_count": load_metadata["unexpected_keys_count"],
        "missing_keys_sample": load_metadata["missing_keys_sample"],
        "unexpected_keys_sample": load_metadata["unexpected_keys_sample"],
        "model_class": load_metadata["model_class"],
        "torch_version": runtime_metadata["torch_version"],
        "open_clip_version": runtime_metadata["open_clip_version"],
        "cuda_available": runtime_metadata["cuda_available"],
        "cuda_device_name": runtime_metadata["cuda_device_name"],
        "original_weights_configured": bool(original_weights_value),
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "source_script": source_script,
        "created_at": utc_now_iso(),
    }
    report_path = safe_write_json(run_dir / "tiny_real_feature_extraction_report.json", report, overwrite=False)
    return report_path, is_valid


class TinyImageSelection:
    def __init__(self, image_paths: list[Path], labels: list[int], class_to_idx: dict[str, int], split_name: str) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.split_name = split_name


def collect_tiny_images(
    *,
    dataset: str,
    split_path: Path | None,
    split_section: str,
    image_list_path: Path | None,
    dataset_root: Path | None,
    max_images: int,
) -> TinyImageSelection:
    if split_path is not None:
        split = read_json(split_path)
        rows = split.get(split_section)
        if not isinstance(rows, list):
            raise ValueError(f"split section {split_section!r} is missing or not a list")
        class_to_idx = split.get("class_to_idx")
        if not isinstance(class_to_idx, dict) or not class_to_idx:
            class_to_idx = {"preflight_class": 0}
        paths: list[Path] = []
        labels: list[int] = []
        for row in rows[:max_images]:
            if not isinstance(row, dict) or "path" not in row:
                raise ValueError("split entries must be objects with a path field")
            path = Path(str(row["path"]))
            paths.append(resolve_explicit_image_path(path, dataset_root, split_path.parent))
            labels.append(int(row.get("label", 0)))
        return TinyImageSelection(paths, labels, {str(key): int(value) for key, value in class_to_idx.items()}, split_section)

    if image_list_path is None:
        raise ValueError("image_list_path is required when split_path is not provided")
    paths = []
    for line in image_list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            paths.append(resolve_explicit_image_path(Path(stripped), dataset_root, image_list_path.parent))
        if len(paths) >= max_images:
            break
    return TinyImageSelection(paths, [0 for _ in paths], {"preflight_class": 0}, f"{dataset}_image_list")


def resolve_explicit_image_path(path: Path, dataset_root: Path | None, fallback_root: Path) -> Path:
    if path.is_absolute():
        return path
    if dataset_root is not None:
        return dataset_root / path
    return fallback_root / path


def output_targets_formal_features_dir(output_dir: Path) -> bool:
    parts = output_dir.parts
    return any(parts[index : index + 2] == ("outputs", "features") for index in range(len(parts) - 1))


def output_targets_allowed_tiny_dir(output_dir: Path) -> bool:
    parts = output_dir.parts
    allowed = {("outputs", "preflight"), ("outputs", "features_tiny_preflight")}
    return any(parts[index : index + 2] in allowed for index in range(len(parts) - 1))


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique tiny real feature extraction directory under {base}")


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
