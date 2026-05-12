from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from scripts.check_adapter_input_preflight import (
    make_split_request,
    resolve_cache_path as resolve_adapter_cache_path,
    resolve_manifest_entries as resolve_adapter_manifest_entries,
    select_shot_support,
)
from scripts.run_zero_shot import (
    load_image_caches_from_manifest,
    load_text_feature_cache,
    prediction_csv_rows,
    validate_image_cache_for_evaluation,
    validate_text_cache_for_evaluation,
)
from src.baselines.proto_adapter import ProtoAdapter
from src.baselines.tip_adapter import TipAdapter
from src.config.config_loader import save_config_snapshot
from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import FeatureCache, load_feature_cache, make_fake_feature_cache, to_labels
from src.logging.experiment_logger import create_unique_run_dir, is_paper_allowed
from src.logging.result_schema import validate_metadata_schema, validate_metrics_schema
from src.logging.system_info import get_system_info
from src.utils.features import argmax_rows
from src.utils.io import read_json, safe_write_csv, write_json_no_overwrite
from src.utils.seed import set_seed
from src.utils.timing import utc_now_iso


SERVER_RUN_MODES = {"server_full", "server_ablation", "server_benchmark"}
SUPPORTED_METHODS = {"tip_adapter", "proto_adapter"}


def add_cached_adapter_args(parser: argparse.ArgumentParser, *, default_config: str, default_method: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--env-config", default="configs/env/local_wsl.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default=default_method)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="", help="Legacy CLI option; use --shot-split for cached adapter mode.")
    parser.add_argument("--feature-cache", default="", help="Legacy single-cache option; cached adapter mode uses --manifest.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--base-split", default="")
    parser.add_argument("--shot-split", default="")
    parser.add_argument("--text-feature-cache", default="")
    parser.add_argument("--adapter-input-plan", default="")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution-env", default=None)
    parser.add_argument("--run-mode", default=None)
    parser.add_argument("--preflight-report", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--allow-paper-result", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--override", action="append", default=[])


def run_cached_training_free_adapter_evaluation(
    *,
    method_name: str,
    config: dict[str, Any],
    config_path: str | Path,
    dataset: str,
    backbone: str,
    shot: int,
    seed: int,
    manifest_path: str | Path | None,
    base_split: str,
    shot_split: str,
    text_feature_cache_path: str | Path | None,
    adapter_input_plan: str | Path | None,
    eval_splits: list[str],
    output_dir: str | Path,
    device: str,
    execution_env: str,
    run_mode: str,
    preflight_report: str | Path | None,
    dry_run: bool,
    max_samples: int | None,
    alpha: float,
    beta: float = 1.0,
    temperature: float = 1.0,
    save_predictions: bool = False,
    allow_paper_result: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported cached adapter method: {method_name}")
    if execution_env == "local_wsl" and device != "cpu":
        raise ValueError("Local WSL cached adapter runs must use --device cpu.")

    set_seed(seed, deterministic=True)
    output_root = Path(output_dir)
    ensure_results_output_allowed(output_root)
    eval_splits = list(dict.fromkeys(eval_splits))
    if not eval_splits:
        raise ValueError("--eval-splits must contain at least one split")

    uses_real_cache_inputs = bool(manifest_path or text_feature_cache_path or adapter_input_plan or preflight_report)
    if uses_real_cache_inputs or not dry_run:
        if not manifest_path:
            raise ValueError("cached adapter evaluation requires --manifest")
        if not base_split:
            raise ValueError("cached adapter evaluation requires --base-split")
        if not shot_split:
            raise ValueError("cached adapter evaluation requires --shot-split")
        if not text_feature_cache_path:
            raise ValueError("cached adapter evaluation requires --text-feature-cache")
        validate_adapter_preflight_report(
            preflight_report=preflight_report,
            dataset=dataset,
            backbone=backbone,
            method_name=method_name,
            base_split=base_split,
            shot_split=shot_split,
            eval_splits=eval_splits,
        )
        validate_adapter_input_plan(
            adapter_input_plan=adapter_input_plan,
            dataset=dataset,
            backbone=backbone,
            method_name=method_name,
            shot=shot,
            shot_split=shot_split,
        )

    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    if dry_run and not manifest_path:
        support_cache, image_caches, image_cache_paths, support_cache_path, effective_base_split, effective_shot_split = (
            make_dry_run_adapter_caches(
                dataset=dataset,
                backbone=backbone,
                seed=seed,
                shot=shot,
                max_samples=max_samples,
                eval_splits=eval_splits,
            )
        )
        text_cache = make_dry_run_text_cache(reference_cache=support_cache, dataset=dataset, backbone=backbone)
    else:
        support_cache, support_cache_path, effective_shot_split = load_support_cache_from_manifest(
            manifest_path=Path(str(manifest_path)),
            dataset=dataset,
            backbone=backbone,
            shot_split=shot_split,
        )
        image_caches, image_cache_paths, effective_base_split = load_image_caches_from_manifest(
            manifest_path=Path(str(manifest_path)),
            dataset=dataset,
            backbone=backbone,
            base_split=base_split,
            eval_splits=eval_splits,
        )
        text_cache = load_text_feature_cache(Path(str(text_feature_cache_path)))

    reference_cache = image_caches[eval_splits[0]]
    text_features, class_names, feature_dim, num_classes = validate_text_cache_for_evaluation(
        text_cache=text_cache,
        image_cache=reference_cache,
        dataset=dataset,
        backbone=backbone,
        base_split=effective_base_split,
        dry_run=dry_run,
    )
    validate_image_cache_for_evaluation(
        support_cache,
        dataset=dataset,
        backbone=backbone,
        class_names=class_names,
        feature_dim=feature_dim,
    )
    validate_support_cache_for_adapter(support_cache, shot=shot, num_classes=num_classes)

    method = make_adapter_method(method_name, alpha=alpha, beta=beta, temperature=temperature, text_features=text_features)
    method.class_names = class_names
    method.fit(support_cache.image_features, support_cache.image_labels)

    per_split_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    total_samples = 0
    inference_time_sec = 0.0
    for split in eval_splits:
        cache = image_caches[split]
        validate_image_cache_for_evaluation(cache, dataset=dataset, backbone=backbone, class_names=class_names, feature_dim=feature_dim)
        split_start = time.perf_counter()
        logits = method.predict_logits(cache.image_features)
        split_elapsed = time.perf_counter() - split_start
        labels = to_labels(cache.image_labels)
        metrics_core = evaluate_logits(logits, labels, class_names=class_names)
        predictions = argmax_rows(logits)
        per_split_metrics[split] = {
            "top1_acc": metrics_core["top1_acc"],
            "num_samples": metrics_core["num_samples"],
            "num_classes": metrics_core["num_classes"],
            "per_class_acc": metrics_core["per_class_acc"],
            "confusion_matrix": metrics_core["confusion_matrix"],
            "inference_time_sec": split_elapsed,
            "images_per_second": float(len(labels) / split_elapsed) if split_elapsed > 0 else 0.0,
        }
        total_samples += len(labels)
        inference_time_sec += split_elapsed
        if save_predictions:
            prediction_rows.extend(prediction_csv_rows(split, cache.image_paths, labels, predictions))

    is_paper_result = is_paper_allowed(execution_env, run_mode, allow_paper_result and run_mode in SERVER_RUN_MODES)
    eligible_for_paper_tables = bool(is_paper_result and run_mode in SERVER_RUN_MODES)
    run_id, run_dir = create_unique_run_dir(output_root, dataset, backbone, method_name, shot, seed)
    run_config = {
        **config,
        "cached_adapter_evaluation": {
            "method": method_name,
            "dataset": dataset,
            "backbone": backbone,
            "base_split": effective_base_split,
            "shot_split": effective_shot_split,
            "shot": shot,
            "seed": seed,
            "eval_splits": eval_splits,
            "manifest_path": str(manifest_path or ""),
            "support_cache_path": support_cache_path,
            "image_cache_paths": image_cache_paths,
            "text_feature_cache_path": str(text_feature_cache_path or ""),
            "adapter_input_plan": str(adapter_input_plan or ""),
            "dry_run": dry_run,
            "save_predictions": save_predictions,
            "alpha": alpha,
            "beta": beta if method_name == "tip_adapter" else None,
            "temperature": temperature,
        },
    }
    config_snapshot_path = save_config_snapshot(run_config, run_dir)
    log_path = run_dir / "log.txt"
    log_path.write_text("Cached training-free adapter evaluation initialized.\n", encoding="utf-8")
    prediction_path = ""
    if save_predictions:
        prediction_path = str(safe_write_csv(run_dir / "predictions.csv", prediction_rows, ["sample_id", "split", "path", "label", "pred", "correct"]))

    end_time = utc_now_iso()
    system_info = get_system_info(device=device)
    top1_by_split = {split: payload["top1_acc"] for split, payload in per_split_metrics.items()}
    primary_split = "test" if "test" in per_split_metrics else eval_splits[-1]
    fake_text = bool(text_cache.get("dry_run", False) or text_cache.get("uses_fake_text_features", False))
    fake_images = bool(
        support_cache.metadata.get("uses_fake_features", False)
        or support_cache.metadata.get("uses_fake_data", False)
        or any(cache.metadata.get("uses_fake_features", False) or cache.metadata.get("uses_fake_data", False) for cache in image_caches.values())
    )
    common_payload = {
        "run_id": run_id,
        "method": method_name,
        "backbone": backbone,
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": device,
        "base_split": effective_base_split,
        "shot_split": effective_shot_split,
        "eval_splits": eval_splits,
        "split_path": effective_shot_split,
        "support_cache_path": support_cache_path,
        "image_cache_paths": image_cache_paths,
        "text_feature_cache_path": str(text_feature_cache_path or ""),
        "adapter_input_plan": str(adapter_input_plan or ""),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "cache_entries": int(method.cache_entries),
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "top1_acc_by_split": top1_by_split,
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": save_predictions,
        "writes_results_raw": True,
    }
    command_text = command or shlex.join(sys.argv)
    metadata = {
        **system_info,
        **common_payload,
        "torch_version": system_info.get("pytorch_version"),
        "command": command_text,
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "server_job_id": None,
        "start_time": start_time,
        "end_time": end_time,
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
    }
    metrics = {
        **system_info,
        **common_payload,
        "torch_version": system_info.get("pytorch_version"),
        "command": command_text,
        "top1_acc": top1_by_split.get(primary_split),
        "per_split": per_split_metrics,
        "num_samples": total_samples,
        "inference_time_sec": inference_time_sec,
        "images_per_second": float(total_samples / inference_time_sec) if inference_time_sec > 0 else 0.0,
        "gpu_memory_mb": None,
        "uses_fake_data": bool(dry_run or fake_images),
        "uses_fake_features": bool(dry_run or fake_images or fake_text),
        "fake_or_dry_run": bool(dry_run or fake_images or fake_text),
        "used_fake_features": bool(dry_run or fake_images or fake_text),
        "is_real_evaluation": not bool(dry_run or fake_images or fake_text),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
        "prediction_path": prediction_path,
        "checkpoint_path": None,
        "start_time": start_time,
        "end_time": end_time,
        "total_time_sec": time.perf_counter() - start_perf,
        "alpha": alpha,
        "beta": beta if method_name == "tip_adapter" else None,
        "temperature": temperature,
    }
    validate_metadata_schema(metadata)
    validate_metrics_schema(metrics)
    metadata_path = write_json_no_overwrite(run_dir / "metadata.json", metadata)
    metrics_path = write_json_no_overwrite(run_dir / "metrics.json", metrics)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"Cached training-free adapter evaluation finished at {end_time}.\n")
    return {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "prediction_path": Path(prediction_path) if prediction_path else None,
        "metrics": metrics,
    }


def make_adapter_method(method_name: str, *, alpha: float, beta: float, temperature: float, text_features: Any):
    if method_name == "tip_adapter":
        return TipAdapter(alpha=alpha, beta=beta, temperature=temperature, text_features=text_features)
    if method_name == "proto_adapter":
        return ProtoAdapter(alpha=alpha, temperature=temperature, text_features=text_features)
    raise ValueError(f"unsupported cached adapter method: {method_name}")


def load_support_cache_from_manifest(
    *, manifest_path: Path, dataset: str, backbone: str, shot_split: str
) -> tuple[FeatureCache, str, str]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_json(manifest_path)
    entries = resolve_adapter_manifest_entries(manifest, manifest_path, errors)
    if errors:
        raise ValueError("; ".join(errors))
    selected_entries = [entry for entry in entries if entry.get("dataset") == dataset and entry.get("backbone") == backbone]
    request = make_split_request(shot_split, dataset)
    entry = select_shot_support(selected_entries, request, warnings)
    if entry is None:
        raise ValueError(f"{shot_split} is missing required support feature cache")
    cache_path = resolve_adapter_cache_path(entry, Path(str(entry.get("summary_path", "."))).parent)
    if cache_path is None or not cache_path.exists():
        raise ValueError(f"support feature cache does not exist: {cache_path}")
    return load_feature_cache(cache_path), str(cache_path), request["split_id"]


def make_dry_run_adapter_caches(
    *, dataset: str, backbone: str, seed: int, shot: int, max_samples: int | None, eval_splits: list[str]
) -> tuple[FeatureCache, dict[str, FeatureCache], dict[str, str], str, str, str]:
    num_classes = 3
    feature_dim = 8
    support_cache = make_fake_feature_cache(
        num_samples=num_classes * shot,
        num_classes=num_classes,
        feature_dim=feature_dim,
        seed=seed,
        split_name="support",
        dataset=dataset,
        backbone=backbone,
    )
    image_caches = {}
    for index, split in enumerate(eval_splits):
        image_caches[split] = make_fake_feature_cache(
            num_samples=max_samples or 12,
            num_classes=num_classes,
            feature_dim=feature_dim,
            seed=seed + index + 100,
            split_name=split,
            dataset=dataset,
            backbone=backbone,
        )
    return support_cache, image_caches, {split: "" for split in eval_splits}, "", "dry_run", f"shot_{shot}_dry_run_seed{seed}"


def make_dry_run_text_cache(*, reference_cache: FeatureCache, dataset: str, backbone: str) -> dict[str, Any]:
    return {
        "text_features": reference_cache.text_features,
        "class_names": reference_cache.class_names,
        "class_to_idx": reference_cache.class_to_idx,
        "prompts": reference_cache.text_prompts or [],
        "dataset": dataset,
        "backbone": backbone,
        "base_split": "dry_run",
        "feature_dim": reference_cache.feature_dim,
        "num_classes": len(reference_cache.class_to_idx),
        "normalize_features": True,
        "source_script": "scripts/cached_adapter_runner.py",
        "created_at": utc_now_iso(),
        "dry_run": True,
        "uses_fake_text_features": True,
        "is_paper_result": False,
    }


def validate_support_cache_for_adapter(cache: FeatureCache, *, shot: int, num_classes: int) -> None:
    labels = to_labels(cache.image_labels)
    expected_total = shot * num_classes
    if len(labels) != expected_total:
        raise ValueError(f"support cache entries={len(labels)} does not equal shot*num_classes={expected_total}")
    counts = {label: labels.count(label) for label in range(num_classes)}
    bad_counts = {label: count for label, count in counts.items() if count != shot}
    if bad_counts:
        raise ValueError(f"support cache per-class counts do not match shot={shot}: {bad_counts}")


def validate_adapter_preflight_report(
    *,
    preflight_report: str | Path | None,
    dataset: str,
    backbone: str,
    method_name: str,
    base_split: str,
    shot_split: str,
    eval_splits: list[str],
) -> None:
    if not preflight_report:
        raise ValueError("cached adapter evaluation requires --preflight-report")
    report = read_json(preflight_report)
    if report.get("is_valid") is not True:
        raise ValueError("adapter input preflight report is not ready")
    if report.get("dataset") != dataset:
        raise ValueError(f"adapter input preflight dataset mismatch, expected {dataset}, found {report.get('dataset')}")
    if report.get("backbone") != backbone:
        raise ValueError(f"adapter input preflight backbone mismatch, expected {backbone}, found {report.get('backbone')}")
    base_summary = find_split_summary(report.get("per_split_summary"), base_split)
    if not base_summary:
        raise ValueError("adapter input preflight report is not ready for requested base split")
    for split in eval_splits:
        ready_key = f"{split}_ready_for_evaluation_input" if split == "test" else f"{split}_ready_for_tuning_input"
        section_ready = bool(base_summary.get("sections", {}).get(split, {}).get("is_ready"))
        if not bool(base_summary.get(ready_key, section_ready)):
            raise ValueError(f"adapter input preflight report is not ready for {split} cache")
    method_summary = report.get("per_method_input_summary", {}).get(method_name, {})
    shot_summary = find_split_summary(method_summary.get("per_shot"), shot_split)
    if not shot_summary or shot_summary.get("method_input_ready") is not True:
        raise ValueError("adapter input preflight report is not ready for requested adapter method and shot split")


def validate_adapter_input_plan(
    *,
    adapter_input_plan: str | Path | None,
    dataset: str,
    backbone: str,
    method_name: str,
    shot: int,
    shot_split: str,
) -> None:
    if not adapter_input_plan:
        raise ValueError("cached adapter evaluation requires --adapter-input-plan")
    plan = read_json(adapter_input_plan)
    if plan.get("source_preflight_is_valid") is not True:
        raise ValueError("adapter input plan is not ready because its source preflight is invalid")
    if plan.get("dataset") != dataset:
        raise ValueError(f"adapter input plan dataset mismatch, expected {dataset}, found {plan.get('dataset')}")
    if plan.get("backbone") != backbone:
        raise ValueError(f"adapter input plan backbone mismatch, expected {backbone}, found {plan.get('backbone')}")
    row = find_plan_row(plan.get("rows"), method_name=method_name, shot=shot, shot_split=shot_split)
    if not row or row.get("is_ready") is not True:
        raise ValueError("adapter input plan is not ready for requested adapter method and shot split")


def find_split_summary(summaries: Any, requested_split: str) -> dict[str, Any] | None:
    if not isinstance(summaries, dict):
        return None
    requested_tokens = split_tokens(requested_split)
    for key, value in summaries.items():
        if requested_tokens & split_tokens(str(key)) and isinstance(value, dict):
            return value
    return None


def find_plan_row(rows: Any, *, method_name: str, shot: int, shot_split: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    requested_tokens = split_tokens(shot_split)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("method") != method_name:
            continue
        if row.get("shot") is not None and int(row["shot"]) != shot:
            continue
        if requested_tokens & split_tokens(str(row.get("shot_split", ""))):
            return row
    return None


def split_tokens(value: str) -> set[str]:
    path = Path(value)
    tokens = {value, path.name, path.stem}
    return {token for token in tokens if token}


def ensure_results_output_allowed(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "outputs" and parts[index + 1] == "preflight":
            raise ValueError("cached adapter evaluation metrics must not be written under outputs/preflight")
