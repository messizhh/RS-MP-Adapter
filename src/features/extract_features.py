from __future__ import annotations

import copy
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.features.feature_cache import FeatureCache, class_names_from_mapping, save_feature_cache
from src.features.image_preprocess import load_rgb_image
from src.models.base_backbone import aggregate_prompt_features, create_backbone, expand_prompts
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


SERVER_REAL_RUN_MODES = {"server_full", "server_benchmark"}


def extract_features_placeholder(*args, **kwargs) -> FeatureCache:
    raise NotImplementedError(
        "Full CLIP/RemoteCLIP/GeoRSCLIP feature extraction is intentionally not implemented in Phase 1D."
    )


def run_dry_run_feature_extraction(
    *,
    dataset: str,
    backbone_name: str,
    backbone_config: dict[str, Any] | None,
    output_dir: str | Path,
    split_path: str | Path | None,
    max_samples: int,
    batch_size: int,
    device: str,
    execution_env: str,
    run_mode: str,
    prompt_templates: list[str] | None = None,
    overwrite: bool = False,
    source_script: str = "scripts/extract_features.py",
) -> dict[str, Any]:
    backbone = create_backbone(backbone_name, backbone_config, dry_run=True, device=device).load_model().eval()
    class_to_idx = {"class_0": 0, "class_1": 1, "class_2": 2}
    class_names = class_names_from_mapping(class_to_idx)
    image_paths = [f"fake://{dataset}/dry_run/sample_{idx:04d}.jpg" for idx in range(max_samples)]
    image_labels = [idx % len(class_to_idx) for idx in range(max_samples)]
    image_features = []
    for start in range(0, max_samples, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        image_features.extend(backbone.encode_images(batch_paths))
    prompts = expand_prompts(class_names, prompt_templates)
    prompt_features = backbone.encode_text(prompts)
    text_features = aggregate_prompt_features(
        prompt_features,
        num_classes=len(class_names),
        templates_per_class=len(prompt_templates or ["a satellite photo of {}."]),
        normalize_features=True,
    )
    cache = FeatureCache(
        image_features=image_features,
        image_labels=image_labels,
        image_paths=image_paths,
        split_name="dry_run",
        class_to_idx=class_to_idx,
        text_features=text_features,
        text_prompts=prompts,
        backbone=backbone.config.name,
        dataset=dataset,
        feature_dim=backbone.get_feature_dim(),
        normalize_features=True,
        created_at=utc_now_iso(),
        source_script=source_script,
        metadata={
            "execution_env": execution_env,
            "run_mode": run_mode,
            "device": device,
            "is_paper_result": False,
            "uses_fake_data": True,
            "uses_fake_features": True,
            "is_real_feature_extraction": False,
            "split_path": str(split_path) if split_path else "",
            "preprocess": backbone.describe_preprocess(),
        },
    )
    cache.validate()
    run_dir = unique_dir(Path(output_dir) / dataset / backbone.config.name / "dry_run")
    cache_path = run_dir / "feature_cache.pt"
    save_feature_cache(cache, cache_path)
    summary = {
        "dataset": dataset,
        "backbone": backbone.config.name,
        "feature_cache_path": str(cache_path),
        "num_images": max_samples,
        "num_classes": len(class_to_idx),
        "feature_dim": backbone.get_feature_dim(),
        "batch_size": batch_size,
        "device": device,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "uses_fake_data": True,
        "uses_fake_features": True,
        "is_real_feature_extraction": False,
        "created_at": utc_now_iso(),
        "source_script": source_script,
    }
    summary_path = safe_write_json(run_dir / "feature_extraction_summary.json", summary, overwrite=overwrite)
    return {"cache": cache, "cache_path": cache_path, "summary_path": summary_path, "run_dir": run_dir}


def run_guarded_real_feature_extraction(
    *,
    dataset: str,
    backbone_name: str,
    backbone_config: dict[str, Any],
    output_dir: str | Path,
    split_path: str | Path,
    split_section: str,
    seed: int | None = None,
    shot: int | None = None,
    dataset_root: str | Path,
    weights_path: str | Path,
    batch_size: int,
    max_samples: int | None = None,
    device: str,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
    is_paper_result_candidate: bool = False,
    source_script: str = "scripts/extract_features.py",
) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    errors = validate_real_extraction_guard(
        split_path=split_path,
        dataset_root=dataset_root,
        weights_path=weights_path,
        execution_env=execution_env,
        run_mode=run_mode,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    if errors:
        raise ValueError("; ".join(errors))

    effective_config = copy.deepcopy(backbone_config)
    backbone_cfg = effective_config.setdefault("backbone", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("backbone config root must contain a backbone mapping")
    backbone_cfg["weights"] = str(weights_path)
    backbone_cfg.pop("pretrained_path", None)

    selected = collect_split_images(
        split_path=Path(split_path),
        split_section=split_section,
        dataset_root=Path(dataset_root),
    )
    split_metadata = infer_split_metadata(
        split_path=Path(split_path),
        split_section=split_section,
        explicit_seed=seed,
        explicit_shot=shot,
    )
    image_count_before_limit = len(selected.image_paths)
    requested_max_samples = max_samples
    if max_samples is not None:
        selected = selected.limit(max_samples)
    image_count_after_limit = len(selected.image_paths)
    max_samples_applied = requested_max_samples is not None and image_count_after_limit < image_count_before_limit
    if not selected.image_paths:
        raise ValueError(f"split section {split_section!r} produced zero images")
    for image_path in selected.image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"image path does not exist: {image_path}")

    runtime_metadata = collect_runtime_metadata(device)
    model_load_time_sec = 0.0
    preprocess_time_sec = 0.0
    encode_time_sec = 0.0
    save_time_sec = 0.0

    model_load_start = time.perf_counter()
    backbone = create_backbone(backbone_name, effective_config, dry_run=False, device=device)
    backbone.load_model().eval()
    model_load_time_sec = time.perf_counter() - model_load_start
    load_metadata = getattr(backbone, "load_metadata", {})
    checkpoint_loaded = bool(load_metadata.get("checkpoint_loaded", False))
    if not checkpoint_loaded:
        raise RuntimeError("server real feature extraction requires checkpoint_loaded=true")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("server real feature extraction requires torch") from exc

    image_size = image_size_from_backbone_config(backbone_cfg)
    feature_batches = []
    preprocess_start = time.perf_counter()
    encode_total = 0.0
    for start in range(0, len(selected.image_paths), batch_size):
        batch_paths = selected.image_paths[start : start + batch_size]
        tensors = []
        for image_path in batch_paths:
            image = load_rgb_image(image_path, image_size=image_size)
            tensors.append(pil_to_clip_tensor(image, device=device))
        image_batch = torch.cat(tensors, dim=0)
        preprocess_time_sec += time.perf_counter() - preprocess_start
        encode_start = time.perf_counter()
        feature_batches.append(backbone.encode_image_preflight(image_batch).detach().cpu())
        encode_total += time.perf_counter() - encode_start
        preprocess_start = time.perf_counter()
    encode_time_sec = encode_total
    features = torch.cat(feature_batches, dim=0)
    feature_shape = list(features.shape)
    norms = features.norm(dim=1).detach().cpu()
    feature_norm_stats = {
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
        "mean": float(norms.mean().item()),
    }
    if not bool(features.isfinite().all().detach().cpu().item()):
        raise RuntimeError("image features contain non-finite values")
    if not all(math.isfinite(value) for value in feature_norm_stats.values()):
        raise RuntimeError("image feature norm stats are not finite")

    end_time = utc_now_iso()
    is_limited_real_extraction = requested_max_samples is not None
    is_full_split_extraction = not is_limited_real_extraction
    effective_paper_candidate = bool(is_paper_result_candidate) and is_full_split_extraction
    load_report_fields = checkpoint_report_fields(load_metadata, "cli_override")
    metadata = {
        "git_commit": git_commit(),
        "command": command or shell_join(sys.argv),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "is_paper_result_candidate": effective_paper_candidate,
        "eligible_for_paper_tables": False,
        "dataset": dataset,
        "backbone": backbone_name,
        **split_metadata,
        "split_path": str(split_path),
        "split_section": split_section,
        "image_count": image_count_after_limit,
        "max_samples": requested_max_samples,
        "requested_max_samples": requested_max_samples,
        "image_count_before_limit": image_count_before_limit,
        "image_count_after_limit": image_count_after_limit,
        "max_samples_applied": max_samples_applied,
        "feature_shape": feature_shape,
        "feature_norm_stats": feature_norm_stats,
        "weights_source": "cli_override",
        "checkpoint_loaded": checkpoint_loaded,
        **load_report_fields,
        "start_time": start_time,
        "end_time": end_time,
        "device": device,
        "torch_version": runtime_metadata["torch_version"],
        "open_clip_version": runtime_metadata["open_clip_version"],
        "checkpoint_num_tensors": int(load_metadata.get("checkpoint_num_tensors", 0)),
        "checkpoint_load_mode": load_metadata.get("checkpoint_load_mode", ""),
        "missing_keys_count": int(load_metadata.get("missing_keys_count", 0)),
        "unexpected_keys_count": int(load_metadata.get("unexpected_keys_count", 0)),
        "extracts_text_features": False,
        "saves_predictions": False,
        "saves_logits": False,
        "trains_model": False,
        "evaluates_model": False,
        "downloads_weights": False,
        "is_real_feature_extraction": True,
        "is_limited_real_extraction": is_limited_real_extraction,
        "is_full_split_extraction": is_full_split_extraction,
        "is_full_feature_extraction": is_full_split_extraction,
        "model_load_time_sec": model_load_time_sec,
        "preprocess_time_sec": preprocess_time_sec,
        "encode_time_sec": encode_time_sec,
        "total_time_sec": time.perf_counter() - start_perf,
    }
    run_dir = unique_dir(Path(output_dir) / dataset / backbone_name / split_section)
    cache_path = run_dir / "feature_cache.pt"
    save_start = time.perf_counter()
    cache = FeatureCache(
        image_features=features,
        image_labels=torch.tensor(selected.labels, dtype=torch.long),
        image_paths=[str(path) for path in selected.image_paths],
        split_name=split_section,
        class_to_idx=selected.class_to_idx,
        text_features=None,
        text_prompts=None,
        backbone=backbone_name,
        dataset=dataset,
        feature_dim=int(features.shape[1]),
        normalize_features=bool(backbone_cfg.get("normalize_features", True)),
        created_at=utc_now_iso(),
        source_script=source_script,
        metadata=metadata,
    )
    save_feature_cache(cache, cache_path)
    save_time_sec = time.perf_counter() - save_start
    summary = {
        **metadata,
        "feature_cache_path": str(cache_path),
        "run_dir": str(run_dir),
        "save_time_sec": save_time_sec,
        "source_script": source_script,
    }
    summary_path = safe_write_json(run_dir / "feature_extraction_summary.json", summary, overwrite=False)
    return {"cache": cache, "cache_path": cache_path, "summary_path": summary_path, "run_dir": run_dir}


def save_fake_features_for_smoke(path: str | Path, seed: int = 1) -> Path:
    from src.features.feature_cache import make_fake_feature_cache

    cache = make_fake_feature_cache(seed=seed)
    return save_feature_cache(cache, path)


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique feature extraction directory under {base}")


class SplitImageSelection:
    def __init__(self, image_paths: list[Path], labels: list[int], class_to_idx: dict[str, int]) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx

    def limit(self, max_samples: int) -> "SplitImageSelection":
        return SplitImageSelection(self.image_paths[:max_samples], self.labels[:max_samples], self.class_to_idx)


def validate_real_extraction_guard(
    *,
    split_path: str | Path | None,
    dataset_root: str | Path | None,
    weights_path: str | Path | None,
    execution_env: str,
    run_mode: str,
    batch_size: int,
    max_samples: int | None = None,
) -> list[str]:
    errors = []
    if execution_env != "remote_server":
        errors.append("real feature extraction requires --execution-env remote_server")
    if run_mode not in SERVER_REAL_RUN_MODES:
        errors.append(f"real feature extraction requires --run-mode in {sorted(SERVER_REAL_RUN_MODES)}")
    if not split_path:
        errors.append("real feature extraction requires an explicit --split path")
    elif not Path(split_path).exists():
        errors.append(f"split path does not exist: {split_path}")
    if not dataset_root:
        errors.append("real feature extraction requires an explicit --dataset-root or dataset config root")
    elif not Path(dataset_root).exists():
        errors.append(f"dataset root does not exist: {dataset_root}")
    if not weights_path:
        errors.append("real feature extraction requires an explicit --weights-path")
    elif not Path(weights_path).exists():
        errors.append(f"weights path does not exist: {weights_path}")
    if batch_size <= 0:
        errors.append("--batch-size must be positive")
    if max_samples is not None and max_samples <= 0:
        errors.append("--max-samples must be positive when provided")
    return errors


def collect_split_images(*, split_path: Path, split_section: str, dataset_root: Path) -> SplitImageSelection:
    split = read_json(split_path)
    rows = split.get(split_section)
    if not isinstance(rows, list):
        raise ValueError(f"split section {split_section!r} is missing or not a list")
    class_to_idx = split.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("split file must contain a non-empty class_to_idx mapping")
    image_paths = []
    labels = []
    for row in rows:
        if not isinstance(row, dict) or "path" not in row:
            raise ValueError("split entries must be objects with a path field")
        image_path = Path(str(row["path"]))
        image_paths.append(image_path if image_path.is_absolute() else dataset_root / image_path)
        labels.append(int(row.get("label", 0)))
    return SplitImageSelection(image_paths, labels, {str(key): int(value) for key, value in class_to_idx.items()})


def infer_split_metadata(
    *,
    split_path: Path,
    split_section: str,
    explicit_seed: int | None,
    explicit_shot: int | None,
) -> dict[str, Any]:
    split_id = canonical_split_id(split_path)
    split_file_stem = split_path.stem
    seed = explicit_seed if explicit_seed is not None else infer_seed_from_text(split_file_stem)
    shot = explicit_shot if explicit_shot is not None else infer_shot_from_text(split_file_stem)
    if split_section in {"train", "val", "test"}:
        shot = None
    return {
        "seed": seed,
        "shot": shot,
        "split": str(split_path),
        "split_id": split_id,
        "split_name": split_id,
        "split_file_stem": split_file_stem,
        "base_split": split_id if split_id.startswith("base_") else None,
    }


def canonical_split_id(split_path: Path) -> str:
    stem = split_path.stem
    match = re.fullmatch(r"base_split_seed(\d+)", stem)
    if match:
        return f"base_seed{match.group(1)}"
    return stem


def infer_seed_from_text(value: str) -> int | None:
    match = re.search(r"seed(\d+)", value)
    return int(match.group(1)) if match else None


def infer_shot_from_text(value: str) -> int | None:
    match = re.search(r"shot_(\d+)", value)
    return int(match.group(1)) if match else None


def image_size_from_backbone_config(backbone_config: dict[str, Any]) -> int:
    image_size = backbone_config.get("image_size", 224)
    if isinstance(image_size, (list, tuple)):
        return int(image_size[0])
    return int(image_size)


def pil_to_clip_tensor(image: Any, *, device: str):
    import numpy as np
    import torch

    array = np.asarray(image).astype("float32") / 255.0
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype="float32")
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype="float32")
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


def collect_runtime_metadata(device: str) -> dict[str, Any]:
    torch_version = None
    cuda_available = False
    cuda_device_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if device.startswith("cuda") and cuda_available:
            cuda_device_name = torch.cuda.get_device_name(device)
    except Exception:
        pass
    open_clip_version = None
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


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


def git_commit() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception:
        return ""
