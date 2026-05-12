#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_cache import load_feature_cache, shape_of_1d, shape_of_2d
from src.logging.system_info import git_commit_hash
from src.prototypes.prototype_builder import PrototypeBuilder
from src.utils.features import to_labels, to_rows
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


SUPPORTED_PROTOTYPE_INITS = {"mean", "random_group_mean", "medoid", "kmeans"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only RS-CPC prototype construction shape preflight.")
    parser.add_argument("--adapter-input-plan", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--prototype-inits", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_rs_cpc_prototype_preflight(
        adapter_input_plan_path=args.adapter_input_plan,
        preflight_report_path=args.preflight_report,
        prototype_inits=args.prototype_inits,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"rs_cpc_prototype_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_rs_cpc_prototype_preflight(
    *,
    adapter_input_plan_path: str | Path,
    preflight_report_path: str | Path,
    prototype_inits: list[str],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> tuple[Path, bool]:
    errors: list[str] = []
    warnings: list[str] = []
    plan_path = Path(adapter_input_plan_path)
    report_path = Path(preflight_report_path)
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)

    adapter_plan = read_json(plan_path)
    preflight_report = read_json(report_path)
    dataset = str(adapter_plan.get("dataset", preflight_report.get("dataset", "unknown_dataset")))
    backbone = str(adapter_plan.get("backbone", preflight_report.get("backbone", "unknown_backbone")))
    seed = str(adapter_plan.get("seed") or infer_seed(preflight_report))
    seed_int = seed_to_int(seed)
    num_classes = int_or_none(adapter_plan.get("num_classes")) or int_or_none(preflight_report.get("num_classes"))
    feature_dim = int_or_none(adapter_plan.get("feature_dim")) or int_or_none(preflight_report.get("feature_dim"))
    support_cache_by_split = support_cache_lookup(preflight_report)

    requested_inits = normalize_prototype_inits(prototype_inits, warnings)
    plan_rows = adapter_plan.get("rows")
    if not isinstance(plan_rows, list):
        errors.append("adapter input plan must contain a rows list")
        plan_rows = []

    checked_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    per_combination_summary: list[dict[str, Any]] = []
    prototype_init_summaries = {
        init_mode: {
            "requested": True,
            "supported": init_mode in {"mean", "random_group_mean", "medoid"},
            "checked": 0,
            "skipped": 0,
        }
        for init_mode in requested_inits
    }

    cache_cache: dict[str, Any] = {}
    for row in plan_rows:
        if not isinstance(row, dict) or row.get("method") != "rs_cpc":
            continue
        shot_split = str(row.get("shot_split", ""))
        shot = int_or_none(row.get("shot"))
        candidate_m = int_or_none(row.get("candidate_M"))
        row_ready = bool(row.get("is_ready"))
        if not row_ready:
            skipped_rows.append(skip_row(row, None, "adapter_input_plan_row_not_ready"))
            continue
        if shot is None or candidate_m is None:
            skipped_rows.append(skip_row(row, None, "missing_shot_or_candidate_M"))
            continue
        if candidate_m > shot:
            skipped_rows.append(skip_row(row, None, "candidate_M_exceeds_shot_from_plan"))
            warnings.append(f"{shot_split}: ready plan row has M={candidate_m} > shot={shot}; skipped")
            continue

        cache_path = support_cache_by_split.get(shot_split)
        if not cache_path:
            errors.append(f"{shot_split}: support cache path not found in source preflight report")
            skipped_rows.append(skip_row(row, None, "support_cache_path_missing"))
            continue
        if cache_path not in cache_cache:
            try:
                cache_cache[cache_path] = load_feature_cache(cache_path)
            except Exception as exc:
                errors.append(f"{shot_split}: failed to load support feature cache {cache_path}: {exc}")
                skipped_rows.append(skip_row(row, None, "support_cache_load_failed"))
                continue
        cache = cache_cache[cache_path]

        cache_errors = validate_support_cache(
            cache=cache,
            shot_split=shot_split,
            shot=shot,
            candidate_m=candidate_m,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        if cache_errors:
            errors.extend(cache_errors)
            skipped_rows.append(skip_row(row, None, "support_cache_shape_check_failed"))
            continue

        for init_mode in requested_inits:
            if init_mode not in SUPPORTED_PROTOTYPE_INITS:
                skipped_rows.append(skip_row(row, init_mode, "unsupported_prototype_init"))
                prototype_init_summaries[init_mode]["skipped"] += 1
                warnings.append(f"{init_mode}: unsupported prototype init requested; skipped")
                continue
            if init_mode == "kmeans":
                skipped_rows.append(skip_row(row, init_mode, "kmeans_not_implemented_for_preflight"))
                prototype_init_summaries[init_mode]["skipped"] += 1
                warnings.append("kmeans is reserved but not implemented for prototype preflight; skipped")
                continue
            if init_mode == "mean" and candidate_m != 1:
                skipped_rows.append(skip_row(row, init_mode, "mean_unsupported_for_M_gt_1"))
                prototype_init_summaries[init_mode]["skipped"] += 1
                warnings.append(f"{shot_split}: mean init supports only M=1 in this preflight; skipped M={candidate_m}")
                continue

            combination = check_prototype_combination(
                cache=cache,
                row=row,
                init_mode=init_mode,
                candidate_m=candidate_m,
                num_classes=num_classes,
                feature_dim=feature_dim,
                seed=seed_int,
            )
            if combination["errors"]:
                errors.extend(f"{shot_split}/{init_mode}/M={candidate_m}: {error}" for error in combination["errors"])
            checked_rows.append(
                {
                    "shot_split": shot_split,
                    "shot": shot,
                    "candidate_M": candidate_m,
                    "prototype_init": init_mode,
                    "support_cache_path": cache_path,
                }
            )
            per_combination_summary.append(combination["summary"])
            prototype_init_summaries[init_mode]["checked"] += 1

    report = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "adapter_input_plan_path": str(plan_path),
        "source_preflight_report": str(report_path),
        "checked_rows": checked_rows,
        "skipped_rows": skipped_rows,
        "prototype_init_summaries": prototype_init_summaries,
        "per_combination_summary": per_combination_summary,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "saves_prototypes": False,
        "loads_model": False,
        "uses_val_for_tuning": False,
        "uses_test_for_evaluation": False,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/check_rs_cpc_prototype_preflight.py",
    }
    report_dir = unique_dir(output_root / f"{dataset}_{backbone}_{seed}")
    output_path = safe_write_json(report_dir / "rs_cpc_prototype_preflight_report.json", report)
    return output_path, bool(report["is_valid"])


def normalize_prototype_inits(values: list[str], warnings: list[str]) -> list[str]:
    normalized = []
    for value in values:
        init_mode = str(value).strip()
        if not init_mode:
            continue
        if init_mode not in normalized:
            normalized.append(init_mode)
    if not normalized:
        warnings.append("no prototype init modes requested")
    return normalized


def support_cache_lookup(preflight_report: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    per_split = preflight_report.get("per_split_summary")
    if not isinstance(per_split, dict):
        return lookup
    for split_key, split_summary in per_split.items():
        if not isinstance(split_summary, dict) or split_summary.get("split_kind") != "shot":
            continue
        support = split_summary.get("support")
        if not isinstance(support, dict) or not support.get("cache_path"):
            continue
        cache_path = str(support["cache_path"])
        keys = {
            str(split_key),
            str(split_summary.get("split_id", "")),
            Path(str(split_summary.get("split_path", ""))).stem,
        }
        for key in keys:
            if key:
                lookup[key] = cache_path
    return lookup


def validate_support_cache(
    *,
    cache: Any,
    shot_split: str,
    shot: int,
    candidate_m: int,
    num_classes: int | None,
    feature_dim: int | None,
) -> list[str]:
    errors: list[str] = []
    feature_shape = shape_of_2d(cache.image_features)
    label_shape = shape_of_1d(cache.image_labels)
    if num_classes is None:
        errors.append(f"{shot_split}: num_classes is missing from plan/preflight report")
    if feature_dim is None:
        errors.append(f"{shot_split}: feature_dim is missing from plan/preflight report")
    if len(feature_shape) != 2:
        errors.append(f"{shot_split}: support features must be 2D")
    if len(label_shape) != 1:
        errors.append(f"{shot_split}: support labels must be 1D")
    if num_classes is not None and len(cache.class_to_idx) != num_classes:
        errors.append(f"{shot_split}: cache num_classes={len(cache.class_to_idx)} does not match report num_classes={num_classes}")
    if feature_dim is not None and len(feature_shape) == 2 and feature_shape[1] != feature_dim:
        errors.append(f"{shot_split}: feature_dim={feature_shape[1]} does not match report feature_dim={feature_dim}")
    if num_classes is not None and len(feature_shape) == 2 and feature_shape[0] != num_classes * shot:
        errors.append(f"{shot_split}: support feature rows={feature_shape[0]} does not equal C*shot={num_classes * shot}")
    if num_classes is not None and len(label_shape) == 1 and label_shape[0] != num_classes * shot:
        errors.append(f"{shot_split}: support label rows={label_shape[0]} does not equal C*shot={num_classes * shot}")
    labels = to_labels(cache.image_labels)
    if num_classes is not None and any(label < 0 or label >= num_classes for label in labels):
        errors.append(f"{shot_split}: support labels outside range 0..{num_classes - 1}")
    counts = Counter(labels)
    if num_classes is not None:
        for label in range(num_classes):
            count = counts.get(label, 0)
            if count < candidate_m:
                errors.append(f"{shot_split}: class {label} has {count} support samples, less than M={candidate_m}")
            if count != shot:
                errors.append(f"{shot_split}: class {label} has {count} support samples, expected shot={shot}")
    return errors


def check_prototype_combination(
    *,
    cache: Any,
    row: dict[str, Any],
    init_mode: str,
    candidate_m: int,
    num_classes: int | None,
    feature_dim: int | None,
    seed: int,
) -> dict[str, Any]:
    errors: list[str] = []
    features = to_rows(cache.image_features)
    labels = to_labels(cache.image_labels)
    prototypes, prototype_labels, _ = PrototypeBuilder(candidate_m, init_mode, seed=seed).build(features, labels)
    prototype_shape = [len(prototypes), len(prototypes[0]) if prototypes else 0]
    prototype_label_shape = [len(prototype_labels)]
    expected_rows = num_classes * candidate_m if num_classes is not None else None
    if expected_rows is not None and prototype_shape[0] != expected_rows:
        errors.append(f"prototype rows={prototype_shape[0]} does not equal C*M={expected_rows}")
    if feature_dim is not None and prototype_shape[1] != feature_dim:
        errors.append(f"prototype feature_dim={prototype_shape[1]} does not match expected feature_dim={feature_dim}")
    if expected_rows is not None and prototype_label_shape[0] != expected_rows:
        errors.append(f"prototype label rows={prototype_label_shape[0]} does not equal C*M={expected_rows}")
    if num_classes is not None and any(label < 0 or label >= num_classes for label in prototype_labels):
        errors.append(f"prototype labels outside range 0..{num_classes - 1}")
    label_counts = Counter(prototype_labels)
    if num_classes is not None:
        for label in range(num_classes):
            if label_counts.get(label, 0) != candidate_m:
                errors.append(f"prototype class {label} count={label_counts.get(label, 0)} does not equal M={candidate_m}")
    if not all_finite(prototypes):
        errors.append("prototypes contain NaN or Inf")
    return {
        "errors": errors,
        "summary": {
            "shot_split": row.get("shot_split"),
            "shot": row.get("shot"),
            "candidate_M": candidate_m,
            "prototype_init": init_mode,
            "support_entries": row.get("support_entries"),
            "expected_cache_entries": row.get("expected_cache_entries"),
            "prototype_shape": prototype_shape,
            "prototype_label_shape": prototype_label_shape,
            "prototype_label_min": min(prototype_labels) if prototype_labels else None,
            "prototype_label_max": max(prototype_labels) if prototype_labels else None,
            "prototype_counts_by_label": {str(label): int(count) for label, count in sorted(label_counts.items())},
            "prototypes_finite": all_finite(prototypes),
            "is_ready": not errors,
        },
    }


def all_finite(rows: list[list[float]]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row)


def skip_row(row: dict[str, Any], prototype_init: str | None, reason: str) -> dict[str, Any]:
    return {
        "shot_split": row.get("shot_split"),
        "shot": row.get("shot"),
        "candidate_M": row.get("candidate_M"),
        "prototype_init": prototype_init,
        "is_ready": False,
        "skip_reason": reason,
    }


def infer_seed(preflight_report: dict[str, Any]) -> str:
    checked_base = preflight_report.get("checked_base_split")
    if isinstance(checked_base, dict) and checked_base.get("seed") is not None:
        return f"seed{checked_base['seed']}"
    for value in [preflight_report.get("manifest_path"), preflight_report.get("source_preflight_report")]:
        if isinstance(value, str):
            match = re.search(r"seed(\d+)", value)
            if match:
                return f"seed{match.group(1)}"
    return "seed_unknown"


def seed_to_int(value: str) -> int:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else 1


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


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("RS-CPC prototype preflight reports must not be written under results/raw")


def unique_dir(base_dir: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique preflight directory under {base_dir}")


if __name__ == "__main__":
    main()
