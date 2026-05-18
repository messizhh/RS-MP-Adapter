#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import pickle
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.features.extract_features import checkpoint_report_fields, collect_runtime_metadata, git_commit
from src.models.base_backbone import BackboneUnavailableError, create_backbone, expand_prompts
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


DEFAULT_PROMPT_TEMPLATES = ["a satellite photo of {}."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone text feature cache.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--base-split", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path, is_valid = run_text_feature_extraction(
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split,
        preflight_report=args.preflight_report,
        backbone_config=args.backbone_config,
        method_config=args.method_config,
        output_dir=args.output_dir,
        weights_path=args.weights_path,
        device=args.device,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        dry_run=args.dry_run,
        command=shlex.join(sys.argv),
    )
    print(f"text_feature_extraction_summary_path={summary_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_text_feature_extraction(
    *,
    dataset: str,
    backbone: str,
    base_split: str,
    preflight_report: str | Path,
    backbone_config: str | Path,
    method_config: str | Path,
    output_dir: str | Path,
    weights_path: str | Path | None,
    device: str,
    execution_env: str,
    run_mode: str,
    dry_run: bool,
    command: str | None = None,
) -> tuple[Path, bool]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    errors: list[str] = []
    warnings: list[str] = []
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    preflight_path = Path(preflight_report)
    backbone_config_path = Path(backbone_config)
    method_config_path = Path(method_config)

    preflight = read_json(preflight_path)
    backbone_cfg = load_yaml_config(backbone_config_path)
    method_cfg = load_yaml_config(method_config_path)
    prepared = prepare_inputs(
        preflight=preflight,
        dataset=dataset,
        backbone=backbone,
        base_split=base_split,
        method_config=method_cfg,
        errors=errors,
        warnings=warnings,
    )
    base_split_id = prepared["base_split_id"]
    normalize_features = bool(
        ((backbone_cfg.get("backbone") if isinstance(backbone_cfg.get("backbone"), dict) else {}) or {}).get(
            "normalize_features",
            True,
        )
    )
    run_dir = unique_dir(text_cache_output_base(output_root, dataset=dataset, backbone=backbone))
    summary_path = run_dir / "text_feature_extraction_summary.json"
    cache_path = run_dir / "text_feature_cache.pt"

    text_features = None
    prompts: list[str] = prepared["prompts"]
    loads_model = False
    real_extracts_text_features = False
    load_metadata: dict[str, Any] = {}
    runtime_metadata = collect_runtime_metadata(device)

    if not dry_run:
        real_mode_errors = validate_real_mode(
            backbone_config=backbone_cfg,
            weights_path=weights_path,
            execution_env=execution_env,
            device=device,
        )
        errors.extend(real_mode_errors)

    if not errors:
        try:
            if dry_run:
                text_features = make_fake_text_features(
                    prompts=prompts,
                    num_classes=prepared["num_classes"],
                    templates_per_class=len(prepared["prompt_templates"]),
                    feature_dim=prepared["feature_dim"],
                    normalize_features=normalize_features,
                    key_prefix=f"{dataset}:{backbone}:{base_split_id}",
                )
            else:
                effective_config = with_weights_override(backbone_cfg, weights_path)
                backbone_obj = create_backbone(backbone, effective_config, dry_run=False, device=device)
                backbone_obj.load_model().eval()
                loads_model = True
                load_metadata = dict(getattr(backbone_obj, "load_metadata", {}))
                if not hasattr(backbone_obj, "encode_text_preflight"):
                    raise BackboneUnavailableError(f"{backbone} does not expose encode_text_preflight")
                encoded_prompts = backbone_obj.encode_text_preflight(prompts)
                text_features = aggregate_text_features(
                    encoded_prompts,
                    num_classes=prepared["num_classes"],
                    templates_per_class=len(prepared["prompt_templates"]),
                    normalize_features=normalize_features,
                )
                real_extracts_text_features = True
        except Exception as exc:
            errors.append(f"text feature extraction failed: {exc}")

    feature_shape = list(text_features.shape) if text_features is not None and hasattr(text_features, "shape") else []
    if text_features is not None:
        expected_shape = [prepared["num_classes"], prepared["feature_dim"]]
        if feature_shape != expected_shape:
            errors.append(f"text_features shape {feature_shape} does not equal expected {expected_shape}")

    cache_written = False
    if text_features is not None and not errors:
        cache = build_text_cache(
            text_features=text_features,
            class_names=prepared["class_names"],
            class_to_idx=prepared["class_to_idx"],
            prompts=prompts,
            prompt_templates=prepared["prompt_templates"],
            dataset=dataset,
            backbone=backbone,
            base_split=base_split_id,
            feature_dim=prepared["feature_dim"],
            normalize_features=normalize_features,
            source_script="scripts/extract_text_features.py",
            execution_env=execution_env,
            run_mode=run_mode,
            command=command or shlex.join(sys.argv),
            dry_run=dry_run,
            uses_fake_text_features=dry_run,
        )
        save_text_cache(cache, cache_path)
        cache_written = True

    summary = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "text_feature_cache_path": str(cache_path) if cache_written else None,
        "run_dir": str(run_dir),
        "feature_shape": feature_shape,
        "num_classes": prepared.get("num_classes"),
        "feature_dim": prepared.get("feature_dim"),
        "class_names": prepared.get("class_names", []),
        "prompt_templates": prepared.get("prompt_templates", []),
        "prompt_aggregation": "mean_per_class_then_l2_normalize" if normalize_features else "mean_per_class",
        "loads_model": loads_model,
        "extracts_text_features": real_extracts_text_features,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "dry_run": dry_run,
        "uses_fake_text_features": dry_run,
        "saves_text_feature_cache": cache_written,
        "preflight_report": str(preflight_path),
        "backbone_config": str(backbone_config_path),
        "method_config": str(method_config_path),
        "preflight_proposed_text_feature_cache_path": preflight.get("proposed_text_feature_cache_path"),
        "device": device,
        "normalize_features": normalize_features,
        "git_commit": git_commit(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/extract_text_features.py",
        "created_at": utc_now_iso(),
        "start_time": start_time,
        "end_time": utc_now_iso(),
        "total_time_sec": time.perf_counter() - start_perf,
        "torch_version": runtime_metadata.get("torch_version"),
        "open_clip_version": runtime_metadata.get("open_clip_version"),
        "cuda_available": runtime_metadata.get("cuda_available"),
        "cuda_device_name": runtime_metadata.get("cuda_device_name"),
        **checkpoint_report_fields(load_metadata, "cli_override" if weights_path else "config"),
    }
    safe_write_json(summary_path, summary, overwrite=False)
    return summary_path, bool(summary["is_valid"])


def prepare_inputs(
    *,
    preflight: dict[str, Any],
    dataset: str,
    backbone: str,
    base_split: str,
    method_config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if preflight.get("dataset") != dataset:
        errors.append(f"preflight dataset mismatch, expected {dataset}, found {preflight.get('dataset')}")
    if preflight.get("backbone") != backbone:
        errors.append(f"preflight backbone mismatch, expected {backbone}, found {preflight.get('backbone')}")
    if preflight.get("class_order_determinable") is False:
        errors.append("preflight report says class order is not determinable")
    class_names = [str(item) for item in preflight.get("class_names", []) if isinstance(item, str)]
    if not class_names:
        errors.append("preflight report must contain non-empty class_names")
    raw_class_to_idx = preflight.get("class_to_idx")
    class_to_idx = parse_class_to_idx(raw_class_to_idx, errors)
    if class_names and class_to_idx:
        ordered = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        if ordered != class_names:
            errors.append("preflight class_names must match class_to_idx index order")
    feature_dim = int_or_none(preflight.get("expected_feature_dim"))
    if feature_dim is None:
        errors.append("preflight report must contain expected_feature_dim")
        feature_dim = 0
    prompt_templates = prompt_templates_from_preflight_or_config(preflight, method_config, warnings, errors)
    prompts = expand_prompts(class_names, prompt_templates) if class_names and prompt_templates else []
    checked_base_split = preflight.get("checked_base_split") if isinstance(preflight.get("checked_base_split"), dict) else {}
    base_split_id = str(checked_base_split.get("split_id") or Path(base_split).stem)
    return {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "feature_dim": feature_dim,
        "prompt_templates": prompt_templates,
        "prompts": prompts,
        "base_split_id": base_split_id,
    }


def parse_class_to_idx(raw_value: Any, errors: list[str]) -> dict[str, int]:
    if not isinstance(raw_value, dict) or not raw_value:
        errors.append("preflight report must contain non-empty class_to_idx")
        return {}
    result: dict[str, int] = {}
    for name, value in raw_value.items():
        parsed = int_or_none(value)
        if parsed is None:
            errors.append("class_to_idx values must be integer class indices")
            return {}
        result[str(name)] = parsed
    values = sorted(result.values())
    if values != list(range(len(values))):
        errors.append("class_to_idx values must be contiguous from 0 to num_classes-1")
    return result


def prompt_templates_from_preflight_or_config(
    preflight: dict[str, Any],
    method_config: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> list[str]:
    raw_templates = preflight.get("prompt_templates")
    source = "preflight"
    if not isinstance(raw_templates, list) or not raw_templates:
        method = method_config.get("method") if isinstance(method_config.get("method"), dict) else {}
        raw_templates = method.get("prompt_templates") if isinstance(method, dict) else None
        source = "method_config"
    if not isinstance(raw_templates, list) or not raw_templates:
        raw_templates = DEFAULT_PROMPT_TEMPLATES
        source = "built_in_default"
    templates = [str(item) for item in raw_templates if isinstance(item, str) and item]
    if source != "preflight":
        warnings.append(f"using prompt templates from {source}")
    if not templates:
        errors.append("prompt templates are missing")
    for template in templates:
        if "{}" not in template:
            errors.append(f"prompt template must contain '{{}}': {template}")
    return templates


def validate_real_mode(
    *,
    backbone_config: dict[str, Any],
    weights_path: str | Path | None,
    execution_env: str,
    device: str,
) -> list[str]:
    errors = []
    if execution_env != "remote_server":
        errors.append("real text feature extraction requires --execution-env remote_server")
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append(f"CUDA device requested but unavailable: {device}")
        except ImportError:
            errors.append("real text feature extraction requires torch")
    backbone_section = backbone_config.get("backbone") if isinstance(backbone_config.get("backbone"), dict) else {}
    configured_weights = weights_path or backbone_section.get("weights") or backbone_section.get("pretrained_path")
    if not configured_weights:
        errors.append("real text feature extraction requires a local weights path in --weights-path or backbone config")
    elif not Path(configured_weights).exists():
        errors.append(f"weights path does not exist: {configured_weights}")
    if bool(backbone_section.get("allow_download", False)):
        errors.append("allow_download must be false; automatic weight downloads are disabled")
    return errors


def with_weights_override(backbone_config: dict[str, Any], weights_path: str | Path | None) -> dict[str, Any]:
    import copy

    updated = copy.deepcopy(backbone_config)
    backbone_section = updated.setdefault("backbone", {})
    if not isinstance(backbone_section, dict):
        raise ValueError("backbone config root must contain a backbone mapping")
    if weights_path is not None:
        backbone_section["weights"] = str(weights_path)
        backbone_section.pop("pretrained_path", None)
    return updated


def make_fake_text_features(
    *,
    prompts: list[str],
    num_classes: int,
    templates_per_class: int,
    feature_dim: int,
    normalize_features: bool,
    key_prefix: str,
) -> Any:
    import torch

    prompt_vectors = []
    for prompt in prompts:
        prompt_vectors.append(fake_vector(f"{key_prefix}:text:{prompt}", feature_dim))
    prompt_tensor = torch.tensor(prompt_vectors, dtype=torch.float32)
    return aggregate_text_features(
        prompt_tensor,
        num_classes=num_classes,
        templates_per_class=templates_per_class,
        normalize_features=normalize_features,
    )


def fake_vector(key: str, feature_dim: int) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return [rng.gauss(0.0, 1.0) for _ in range(feature_dim)]


def aggregate_text_features(
    prompt_features: Any,
    *,
    num_classes: int,
    templates_per_class: int,
    normalize_features: bool,
) -> Any:
    import torch

    if templates_per_class <= 0:
        raise ValueError("templates_per_class must be positive")
    if not hasattr(prompt_features, "shape"):
        prompt_features = torch.tensor(prompt_features, dtype=torch.float32)
    prompt_features = prompt_features.detach().float().cpu()
    if list(prompt_features.shape[:1]) != [num_classes * templates_per_class]:
        raise ValueError("prompt feature count does not match num_classes * templates_per_class")
    if normalize_features:
        prompt_features = torch.nn.functional.normalize(prompt_features, dim=1)
    reshaped = prompt_features.reshape(num_classes, templates_per_class, -1)
    class_features = reshaped.mean(dim=1)
    if normalize_features:
        class_features = torch.nn.functional.normalize(class_features, dim=1)
    return class_features


def build_text_cache(
    *,
    text_features: Any,
    class_names: list[str],
    class_to_idx: dict[str, int],
    prompts: list[str],
    prompt_templates: list[str],
    dataset: str,
    backbone: str,
    base_split: str,
    feature_dim: int,
    normalize_features: bool,
    source_script: str,
    execution_env: str,
    run_mode: str,
    command: str,
    dry_run: bool,
    uses_fake_text_features: bool,
) -> dict[str, Any]:
    created_at = utc_now_iso()
    return {
        "text_features": text_features,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "prompts": prompts,
        "prompt_templates": prompt_templates,
        "dataset": dataset,
        "backbone": backbone,
        "base_split": base_split,
        "feature_dim": feature_dim,
        "num_classes": len(class_names),
        "normalize_features": normalize_features,
        "source_script": source_script,
        "created_at": created_at,
        "git_commit": git_commit(),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "command": command,
        "dry_run": dry_run,
        "uses_fake_text_features": uses_fake_text_features,
        "loads_model": not dry_run,
        "extracts_text_features": not dry_run,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "prompt_aggregation": "mean_per_class_then_l2_normalize" if normalize_features else "mean_per_class",
    }


def save_text_cache(cache: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite text feature cache: {path}")
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    return path


def text_cache_output_base(output_root: Path, *, dataset: str, backbone: str) -> Path:
    return output_root / dataset / backbone / "text"


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique text feature extraction directory under {base}")


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("text feature extraction outputs must not be written under results/raw")


if __name__ == "__main__":
    main()
