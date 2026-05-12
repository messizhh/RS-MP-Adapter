#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


LOCAL_RUN_MODES = {"dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"}
REQUIRED_RUN_FILES = ["metrics.json", "metadata.json", "log.txt"]
METADATA_REQUIRED_FIELDS = [
    "method",
    "dataset",
    "backbone",
    "seed",
    "execution_env",
    "run_mode",
    "is_paper_result",
    "eligible_for_paper_tables",
    "device",
    "git_commit",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "command",
    "start_time",
    "end_time",
    "result_json_path",
    "log_path",
]
ZERO_SHOT_REQUIRED_FIELDS = [
    "base_split",
    "eval_splits",
    "image_cache_paths",
    "text_feature_cache_path",
    "feature_dim",
    "num_classes",
    "computes_logits",
    "computes_accuracy",
    "evaluates_model",
    "trains_model",
    "extracts_features",
    "loads_model",
    "saves_predictions",
    "writes_results_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only result run directory sanity preflight.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-method", required=True)
    parser.add_argument("--expected-dataset", required=True)
    parser.add_argument("--expected-backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_result_run_preflight(
        run_dir=args.run_dir,
        expected_method=args.expected_method,
        expected_dataset=args.expected_dataset,
        expected_backbone=args.expected_backbone,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        command=shlex.join(sys.argv),
    )
    print(f"result_run_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_result_run_preflight(
    *,
    run_dir: str | Path,
    expected_method: str,
    expected_dataset: str,
    expected_backbone: str,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> tuple[Path, bool]:
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    run_path = Path(run_dir)
    checked_files = check_required_files(run_path, errors)
    metadata = read_json_if_present(run_path / "metadata.json", errors)
    metrics = read_json_if_present(run_path / "metrics.json", errors)

    check_metadata(
        metadata=metadata,
        metrics=metrics,
        expected_method=expected_method,
        expected_dataset=expected_dataset,
        expected_backbone=expected_backbone,
        execution_env=execution_env,
        run_mode=run_mode,
        errors=errors,
        warnings=warnings,
    )
    check_zero_shot_fields(metadata=metadata, metrics=metrics, run_dir=run_path, errors=errors, warnings=warnings)
    check_zero_shot_metrics(metrics=metrics, errors=errors)
    paper_filtering_summary = check_paper_filtering(
        metadata=metadata,
        metrics=metrics,
        run_dir=run_path,
        run_mode=run_mode,
        errors=errors,
        recommendations=recommendations,
    )

    metadata_summary = summarize_metadata(metadata)
    metrics_summary = summarize_metrics(metrics)
    seed = metadata.get("seed", metrics.get("seed", "unknown"))
    report = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations,
        "run_dir": str(run_path),
        "checked_files": checked_files,
        "metadata_summary": metadata_summary,
        "metrics_summary": metrics_summary,
        "paper_filtering_summary": paper_filtering_summary,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
        "is_paper_result": False,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/check_result_run_preflight.py",
    }
    destination = unique_dir(
        Path(output_dir) / f"{expected_dataset}_{expected_backbone}_{expected_method}_seed{seed}"
    )
    report_path = safe_write_json(destination / "result_run_preflight_report.json", report)
    return report_path, bool(report["is_valid"])


def check_required_files(run_dir: Path, errors: list[str]) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        checked[name] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            errors.append(f"missing required run file: {path}")
    config_candidates = [run_dir / "config.yaml", run_dir / "config_snapshot.yaml"]
    config_exists = [path for path in config_candidates if path.exists()]
    checked["config"] = {"paths": [str(path) for path in config_candidates], "exists": bool(config_exists)}
    if not config_exists:
        errors.append(f"missing required config snapshot: expected config.yaml or config_snapshot.yaml in {run_dir}")
    return checked


def read_json_if_present(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        errors.append(f"failed to read {path}: {exc}")
        return {}


def check_metadata(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    expected_method: str,
    expected_dataset: str,
    expected_backbone: str,
    execution_env: str,
    run_mode: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    for field in METADATA_REQUIRED_FIELDS:
        if field not in metadata:
            errors.append(f"metadata.json missing required field: {field}")
    for field in ZERO_SHOT_REQUIRED_FIELDS:
        if field not in metadata and field not in metrics:
            errors.append(f"zero-shot run missing required field: {field}")
    for source_name, payload in [("metadata", metadata), ("metrics", metrics)]:
        if payload.get("method") != expected_method:
            errors.append(f"{source_name} method mismatch, expected {expected_method}, found {payload.get('method')}")
        if payload.get("dataset") != expected_dataset:
            errors.append(f"{source_name} dataset mismatch, expected {expected_dataset}, found {payload.get('dataset')}")
        if payload.get("backbone") != expected_backbone:
            errors.append(f"{source_name} backbone mismatch, expected {expected_backbone}, found {payload.get('backbone')}")
    if metadata.get("execution_env") != execution_env:
        warnings.append(f"metadata execution_env={metadata.get('execution_env')} differs from checker input {execution_env}")
    if metadata.get("run_mode") != run_mode:
        warnings.append(f"metadata run_mode={metadata.get('run_mode')} differs from checker input {run_mode}")
    if metadata.get("result_json_path") and Path(str(metadata["result_json_path"])).name != "metrics.json":
        errors.append("metadata result_json_path must point to metrics.json")
    if metadata.get("log_path") and Path(str(metadata["log_path"])).name != "log.txt":
        errors.append("metadata log_path must point to log.txt")


def check_zero_shot_fields(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    merged = {**metadata, **metrics}
    expected_bools = {
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "writes_results_raw": True,
    }
    for field, expected in expected_bools.items():
        if merged.get(field) is not expected:
            errors.append(f"{field} must be {str(expected).lower()}, found {merged.get(field)}")
    predictions_path = run_dir / "predictions.csv"
    saves_predictions = bool(merged.get("saves_predictions", False))
    if saves_predictions and not predictions_path.exists():
        errors.append("saves_predictions=true but predictions.csv is missing")
    if predictions_path.exists() and not saves_predictions:
        warnings.append("predictions.csv exists although saves_predictions=false")
    eval_splits = merged.get("eval_splits")
    if not isinstance(eval_splits, list) or not eval_splits:
        errors.append("eval_splits must be a non-empty list")
    image_cache_paths = merged.get("image_cache_paths")
    if not isinstance(image_cache_paths, dict) or not image_cache_paths:
        errors.append("image_cache_paths must be a non-empty mapping")
    text_feature_cache_path = merged.get("text_feature_cache_path")
    if not isinstance(text_feature_cache_path, str) or not text_feature_cache_path:
        errors.append("text_feature_cache_path must be a non-empty string")
    if int_or_none(merged.get("feature_dim")) is None:
        errors.append("feature_dim must be an integer")
    if int_or_none(merged.get("num_classes")) is None:
        errors.append("num_classes must be an integer")


def check_zero_shot_metrics(metrics: dict[str, Any], errors: list[str]) -> None:
    num_classes = int_or_none(metrics.get("num_classes"))
    top1_by_split = metrics.get("top1_acc_by_split")
    per_split = metrics.get("per_split")
    if not isinstance(top1_by_split, dict):
        errors.append("metrics top1_acc_by_split must be a mapping")
        top1_by_split = {}
    if not isinstance(per_split, dict):
        errors.append("metrics per_split must be a mapping")
        per_split = {}
    for split in ["val", "test"]:
        if split not in top1_by_split:
            errors.append(f"top1_acc_by_split missing split: {split}")
        elif not is_unit_interval(top1_by_split[split]):
            errors.append(f"top1_acc_by_split[{split}] must be in [0, 1], found {top1_by_split[split]}")
        if split not in per_split:
            errors.append(f"per_split missing split: {split}")
            continue
        split_payload = per_split[split]
        if not isinstance(split_payload, dict):
            errors.append(f"per_split[{split}] must be a mapping")
            continue
        split_top1 = split_payload.get("top1_acc")
        if split_top1 is not None and not is_unit_interval(split_top1):
            errors.append(f"per_split[{split}].top1_acc must be in [0, 1], found {split_top1}")
        if num_classes is not None:
            per_class_acc = split_payload.get("per_class_acc")
            if not isinstance(per_class_acc, list) or len(per_class_acc) != num_classes:
                errors.append(f"per_split[{split}].per_class_acc length must equal num_classes={num_classes}")
            matrix = split_payload.get("confusion_matrix")
            if not matrix_has_shape(matrix, num_classes):
                errors.append(f"per_split[{split}].confusion_matrix must have shape [{num_classes}, {num_classes}]")
        if int_or_none(split_payload.get("num_samples")) is None:
            errors.append(f"per_split[{split}].num_samples must be an integer")
    split_sample_total = sum(
        int(per_split[split].get("num_samples", 0))
        for split in per_split
        if isinstance(per_split.get(split), dict) and int_or_none(per_split[split].get("num_samples")) is not None
    )
    total_samples = int_or_none(metrics.get("num_samples"))
    if total_samples is not None and split_sample_total and total_samples != split_sample_total:
        errors.append(f"metrics num_samples={total_samples} does not equal per_split total={split_sample_total}")
    check_cache_entries_by_method(metrics=metrics, num_classes=num_classes, errors=errors)
    if int_or_none(metrics.get("trainable_params")) != 0:
        errors.append("trainable_params must be 0 for zero-shot")
    if float_or_none(metrics.get("training_time_sec")) != 0.0:
        errors.append("training_time_sec must be 0.0 for zero-shot")


def check_cache_entries_by_method(*, metrics: dict[str, Any], num_classes: int | None, errors: list[str]) -> None:
    if num_classes is None:
        return
    method = str(metrics.get("method", ""))
    cache_entries = int_or_none(metrics.get("cache_entries"))
    if method == "zero_shot":
        if cache_entries != num_classes:
            errors.append("zero_shot cache_entries must equal num_classes")
        return
    if method == "proto_adapter":
        if cache_entries != num_classes:
            errors.append("proto_adapter cache_entries must equal num_classes")
        return
    if method == "tip_adapter":
        shot = int_or_none(metrics.get("shot"))
        if shot is None or shot <= 0:
            errors.append("tip_adapter shot must be a positive integer to validate cache_entries")
            return
        if cache_entries != num_classes * shot:
            errors.append("tip_adapter cache_entries must equal num_classes * shot")


def check_paper_filtering(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
    run_mode: str,
    errors: list[str],
    recommendations: list[str],
) -> dict[str, Any]:
    metadata_paper = bool(metadata.get("is_paper_result", False))
    metrics_paper = bool(metrics.get("is_paper_result", False))
    metadata_eligible = bool(metadata.get("eligible_for_paper_tables", False))
    metrics_eligible = bool(metrics.get("eligible_for_paper_tables", False))
    if run_mode in LOCAL_RUN_MODES:
        if metadata_paper or metrics_paper:
            errors.append(f"run_mode={run_mode} must not be marked as is_paper_result=true")
        if metadata_eligible or metrics_eligible:
            errors.append(f"run_mode={run_mode} must not be eligible_for_paper_tables=true")
    if "results/raw" in run_dir.as_posix() and run_mode in LOCAL_RUN_MODES:
        recommendations.append("Keep this local_validation run excluded from paper-facing tables.")
    return {
        "run_mode": run_mode,
        "metadata_is_paper_result": metadata_paper,
        "metrics_is_paper_result": metrics_paper,
        "metadata_eligible_for_paper_tables": metadata_eligible,
        "metrics_eligible_for_paper_tables": metrics_eligible,
        "is_local_or_debug_mode": run_mode in LOCAL_RUN_MODES,
        "results_raw_output": "results/raw" in run_dir.as_posix(),
    }


def summarize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": metadata.get("method"),
        "dataset": metadata.get("dataset"),
        "backbone": metadata.get("backbone"),
        "seed": metadata.get("seed"),
        "execution_env": metadata.get("execution_env"),
        "run_mode": metadata.get("run_mode"),
        "is_paper_result": metadata.get("is_paper_result"),
        "eligible_for_paper_tables": metadata.get("eligible_for_paper_tables"),
        "device": metadata.get("device"),
        "start_time": metadata.get("start_time"),
        "end_time": metadata.get("end_time"),
    }


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": metrics.get("method"),
        "shot": metrics.get("shot"),
        "top1_acc": metrics.get("top1_acc"),
        "top1_acc_by_split": metrics.get("top1_acc_by_split"),
        "num_classes": metrics.get("num_classes"),
        "feature_dim": metrics.get("feature_dim"),
        "num_samples": metrics.get("num_samples"),
        "cache_entries": metrics.get("cache_entries"),
        "trainable_params": metrics.get("trainable_params"),
        "training_time_sec": metrics.get("training_time_sec"),
        "computes_logits": metrics.get("computes_logits"),
        "computes_accuracy": metrics.get("computes_accuracy"),
        "evaluates_model": metrics.get("evaluates_model"),
    }


def is_unit_interval(value: Any) -> bool:
    parsed = float_or_none(value)
    return parsed is not None and 0.0 <= parsed <= 1.0


def matrix_has_shape(value: Any, size: int) -> bool:
    if not isinstance(value, list) or len(value) != size:
        return False
    return all(isinstance(row, list) and len(row) == size for row in value)


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def unique_dir(base_dir: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique result run preflight directory under {base_dir}")


if __name__ == "__main__":
    main()
