#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


EXCLUDED_RUN_MODES = {"dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"}
DEFAULT_REQUIRED_EXECUTION_ENV = "remote_server"
DEFAULT_REQUIRED_RUN_MODE = "server_full"
DEFAULT_OUTPUT_ROOT = "results/tables"
RAW_PAPER_FLAG_NOTE = (
    "Verified analysis table generated from full-matrix whitelist and post-run preflight, "
    "pending final paper-facing inclusion policy."
)
KNOWN_NUM_CLASSES = {
    "eurosat": 10,
    "aid": 30,
    "nwpu_resisc45": 45,
}

RUN_DIR_KEYS = ["run_dir", "run_directory", "result_run_dir"]
METRICS_PATH_KEYS = ["metrics_json_path", "metrics_path", "metrics_json", "result_json_path"]
METADATA_PATH_KEYS = ["metadata_json_path", "metadata_path", "metadata_json"]
PREFLIGHT_REPORT_PATH_KEYS = [
    "report_path",
    "preflight_report_path",
    "result_run_preflight_report_path",
    "post_run_preflight_report_path",
]
PREFLIGHT_EXIT_CODE_KEYS = ["preflight_exit_code", "exit_code", "result_run_preflight_exit_code"]
PREFLIGHT_VALID_KEYS = ["is_valid", "preflight_is_valid", "result_run_preflight_is_valid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export verified analysis tables from a post-run preflight whitelist TSV."
    )
    parser.add_argument("--preflight-summary", required=True, help="post_run_preflight_summary.tsv")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default="", help="Exact output directory. If omitted, a Day 2 timestamp directory is created.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--required-execution-env", default=DEFAULT_REQUIRED_EXECUTION_ENV)
    parser.add_argument("--required-run-mode", default=DEFAULT_REQUIRED_RUN_MODE)
    parser.add_argument("--allow-fake-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = export_verified_tables_from_preflight(
            preflight_summary=args.preflight_summary,
            output_root=args.output_root,
            output_dir=args.output_dir or None,
            dataset=args.dataset or None,
            backbone=args.backbone or None,
            required_execution_env=args.required_execution_env,
            required_run_mode=args.required_run_mode,
            allow_fake_results=args.allow_fake_results,
            command=shlex.join(sys.argv),
        )
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"day2_verified_table_dir={result['output_dir']}")
    print(f"audit_summary_json={result['audit_summary_json']}")
    print(f"audit_summary_md={result['audit_summary_md']}")
    print(f"included_rows={result['num_included']}")
    print(f"excluded_rows={result['num_excluded']}")


def export_verified_tables_from_preflight(
    *,
    preflight_summary: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    dataset: str | None = None,
    backbone: str | None = None,
    required_execution_env: str = DEFAULT_REQUIRED_EXECUTION_ENV,
    required_run_mode: str = DEFAULT_REQUIRED_RUN_MODE,
    allow_fake_results: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    summary_path = Path(preflight_summary)
    summary_bytes = summary_path.read_bytes()
    preflight_rows = read_tsv(summary_path)
    destination = make_output_dir(
        output_root=Path(output_root),
        output_dir=Path(output_dir) if output_dir else None,
        dataset=dataset,
        backbone=backbone,
    )

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, row in enumerate(preflight_rows, start=1):
        candidate = validate_preflight_row(
            row=row,
            row_index=index,
            summary_path=summary_path,
            dataset=dataset,
            backbone=backbone,
            required_execution_env=required_execution_env,
            required_run_mode=required_run_mode,
            allow_fake_results=allow_fake_results,
        )
        if candidate["include"]:
            included.append(candidate["record"])
        else:
            excluded.append(candidate["audit_row"])

    if summary_path.exists() and summary_path.read_bytes() != summary_bytes:
        raise RuntimeError(f"input preflight summary was modified unexpectedly: {summary_path}")

    outputs = write_table_package(
        output_dir=destination,
        preflight_summary=summary_path,
        preflight_rows=preflight_rows,
        included=included,
        excluded=excluded,
        dataset=dataset,
        backbone=backbone,
        required_execution_env=required_execution_env,
        required_run_mode=required_run_mode,
        allow_fake_results=allow_fake_results,
        command=command or shlex.join(sys.argv),
    )
    return {
        "output_dir": str(destination),
        "num_preflight_rows": len(preflight_rows),
        "num_included": len(included),
        "num_excluded": len(excluded),
        **outputs,
    }


def validate_preflight_row(
    *,
    row: dict[str, str],
    row_index: int,
    summary_path: Path,
    dataset: str | None,
    backbone: str | None,
    required_execution_env: str,
    required_run_mode: str,
    allow_fake_results: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    base_dir = summary_path.parent

    run_dir = resolve_path_from_row(row, RUN_DIR_KEYS, base_dir)
    metrics_path = resolve_path_from_row(row, METRICS_PATH_KEYS, base_dir)
    metadata_path = resolve_path_from_row(row, METADATA_PATH_KEYS, base_dir)
    report_path = resolve_path_from_row(row, PREFLIGHT_REPORT_PATH_KEYS, base_dir)
    preflight_exit_code = normalize_exit_code(first_value(row, PREFLIGHT_EXIT_CODE_KEYS))
    preflight_is_valid = normalize_bool(first_value(row, PREFLIGHT_VALID_KEYS))

    if run_dir is None:
        errors.append("missing run_dir in preflight summary row")
    if run_dir is not None:
        metrics_path = metrics_path or run_dir / "metrics.json"
        metadata_path = metadata_path or run_dir / "metadata.json"
    if metrics_path is None:
        errors.append("missing metrics.json path")
    if metadata_path is None:
        errors.append("missing metadata.json path")
    if report_path is None:
        errors.append("missing post-run preflight report_path")
    if preflight_exit_code != 0:
        errors.append(f"preflight_exit_code is not 0: {preflight_exit_code}")
    if preflight_is_valid is not True:
        errors.append(f"preflight is_valid is not true: {preflight_is_valid}")

    metrics = read_json_for_audit(metrics_path, "metrics.json", errors)
    metadata = read_json_for_audit(metadata_path, "metadata.json", errors)
    report = read_json_for_audit(report_path, "post-run preflight report", errors)

    if report and report.get("is_valid") is not True:
        errors.append(f"post-run preflight report is_valid is not true: {report.get('is_valid')}")
    if run_dir is not None and report.get("run_dir"):
        report_run_dir = resolve_path(str(report["run_dir"]), base_dir)
        if report_run_dir.resolve() != run_dir.resolve():
            errors.append(f"preflight report run_dir mismatch: {report_run_dir} != {run_dir}")

    metrics_env = str(metrics.get("execution_env", ""))
    metadata_env = str(metadata.get("execution_env", ""))
    metrics_mode = str(metrics.get("run_mode", ""))
    metadata_mode = str(metadata.get("run_mode", ""))
    for source_name, value in [("metrics.run_mode", metrics_mode), ("metadata.run_mode", metadata_mode)]:
        if value in EXCLUDED_RUN_MODES:
            errors.append(f"{source_name} is explicitly excluded: {value}")
    for source_name, value in [("metrics.execution_env", metrics_env), ("metadata.execution_env", metadata_env)]:
        if value != required_execution_env:
            errors.append(f"{source_name} must be {required_execution_env}, found {value}")
    for source_name, value in [("metrics.run_mode", metrics_mode), ("metadata.run_mode", metadata_mode)]:
        if value != required_run_mode:
            errors.append(f"{source_name} must be {required_run_mode}, found {value}")

    if dataset:
        for source_name, payload in [("metrics", metrics), ("metadata", metadata)]:
            if payload.get("dataset") != dataset:
                errors.append(f"{source_name}.dataset must be {dataset}, found {payload.get('dataset')}")
    if backbone:
        for source_name, payload in [("metrics", metrics), ("metadata", metadata)]:
            if payload.get("backbone") != backbone:
                errors.append(f"{source_name}.backbone must be {backbone}, found {payload.get('backbone')}")

    if not allow_fake_results:
        for source_name, payload in [("metrics", metrics), ("metadata", metadata)]:
            for flag in ["fake_or_dry_run", "uses_fake_data", "uses_fake_features", "used_fake_features"]:
                if payload.get(flag) is True:
                    errors.append(f"{source_name}.{flag}=true requires --allow-fake-results")

    normalized, normalize_errors, normalize_warnings = normalize_included_record(
        row=row,
        row_index=row_index,
        run_dir=run_dir,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        report_path=report_path,
        preflight_exit_code=preflight_exit_code,
        preflight_is_valid=preflight_is_valid,
        metrics=metrics,
        metadata=metadata,
    )
    errors.extend(normalize_errors)
    warnings.extend(normalize_warnings)

    audit_row = {
        "preflight_summary_row": row_index,
        "include": False,
        "exclusion_reasons": "; ".join(errors),
        "warnings": "; ".join(warnings),
        "run_dir": path_to_str(run_dir),
        "metrics_json_path": path_to_str(metrics_path),
        "metadata_json_path": path_to_str(metadata_path),
        "preflight_report_path": path_to_str(report_path),
        "preflight_exit_code": preflight_exit_code if preflight_exit_code is not None else "",
        "preflight_is_valid": preflight_is_valid if preflight_is_valid is not None else "",
        "metrics_execution_env": metrics_env,
        "metadata_execution_env": metadata_env,
        "metrics_run_mode": metrics_mode,
        "metadata_run_mode": metadata_mode,
        "raw_metrics_is_paper_result": metrics.get("is_paper_result", ""),
        "raw_metadata_is_paper_result": metadata.get("is_paper_result", ""),
        "raw_metrics_eligible_for_paper_tables": metrics.get("eligible_for_paper_tables", ""),
        "raw_metadata_eligible_for_paper_tables": metadata.get("eligible_for_paper_tables", ""),
    }
    if errors:
        return {"include": False, "audit_row": audit_row}

    normalized["warnings"] = "; ".join(warnings)
    return {"include": True, "record": normalized}


def normalize_included_record(
    *,
    row: dict[str, str],
    row_index: int,
    run_dir: Path | None,
    metrics_path: Path | None,
    metadata_path: Path | None,
    report_path: Path | None,
    preflight_exit_code: int | None,
    preflight_is_valid: bool | None,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    method = str(metrics.get("method", metadata.get("method", "")))
    dataset = str(metrics.get("dataset", metadata.get("dataset", "")))
    backbone = str(metrics.get("backbone", metadata.get("backbone", "")))
    seed = value_or(metrics.get("seed"), metadata.get("seed"), "")
    shot = value_or(metrics.get("shot"), metadata.get("shot"), "")
    top1_acc, top1_acc_source = extract_top1_acc(metrics)
    if top1_acc is None:
        errors.append("missing top1_acc in metrics")

    m_value = ""
    prototype_init = ""
    m_sources = ""
    prototype_sources = ""
    if method == "rs_cpc":
        m_result = recover_rs_cpc_m(metrics=metrics, metadata=metadata, run_dir=run_dir)
        if m_result["error"]:
            errors.append(m_result["error"])
        m_value = m_result["value"] if m_result["value"] is not None else ""
        m_sources = "; ".join(m_result["sources"])
        prototype_result = recover_prototype_init(metrics=metrics, metadata=metadata, run_dir=run_dir)
        if prototype_result["error"]:
            errors.append(prototype_result["error"])
        prototype_init = prototype_result["value"] or ""
        prototype_sources = "; ".join(prototype_result["sources"])

    method_variant = make_method_variant(method, m_value, prototype_init)
    num_classes = int_or_empty(value_or(metrics.get("num_classes"), metadata.get("num_classes"), KNOWN_NUM_CLASSES.get(dataset)))
    cache_entries = value_or(metrics.get("cache_entries"), "")
    compression_ratio = value_or(metrics.get("compression_ratio"), "")
    if compression_ratio == "" and method == "rs_cpc":
        compression_ratio = derive_compression_ratio(metrics)

    record = {
        "preflight_summary_row": row_index,
        "include": True,
        "dataset": dataset,
        "shot": shot,
        "shot_label": shot_label(method, shot),
        "backbone": backbone,
        "method": method,
        "method_variant": method_variant,
        "seed": seed,
        "top1_acc": top1_acc if top1_acc is not None else "",
        "top1_acc_source": top1_acc_source,
        "cache_entries": cache_entries,
        "trainable_params": value_or(metrics.get("trainable_params"), ""),
        "training_time_sec": value_or(metrics.get("training_time_sec"), ""),
        "inference_time_sec": value_or(metrics.get("inference_time_sec"), ""),
        "images_per_second": value_or(metrics.get("images_per_second"), ""),
        "gpu_memory_mb": value_or(metrics.get("gpu_memory_mb"), ""),
        "num_samples": value_or(metrics.get("num_samples"), ""),
        "num_classes": num_classes,
        "num_prototypes_per_class": m_value,
        "prototype_init": prototype_init,
        "rs_cpc_m_sources": m_sources,
        "prototype_init_sources": prototype_sources,
        "compression_ratio": compression_ratio,
        "original_cache_entries": value_or(metrics.get("original_cache_entries"), ""),
        "compressed_cache_entries": value_or(metrics.get("compressed_cache_entries"), ""),
        "execution_env": metrics.get("execution_env", ""),
        "run_mode": metrics.get("run_mode", ""),
        "raw_metrics_is_paper_result": metrics.get("is_paper_result", ""),
        "raw_metadata_is_paper_result": metadata.get("is_paper_result", ""),
        "raw_metrics_eligible_for_paper_tables": metrics.get("eligible_for_paper_tables", ""),
        "raw_metadata_eligible_for_paper_tables": metadata.get("eligible_for_paper_tables", ""),
        "verified_analysis_table": True,
        "paper_facing_policy_status": "pending_final_policy",
        "policy_note": RAW_PAPER_FLAG_NOTE,
        "run_id": value_or(metrics.get("run_id"), metadata.get("run_id"), ""),
        "run_dir": path_to_str(run_dir),
        "metrics_json_path": path_to_str(metrics_path),
        "metadata_json_path": path_to_str(metadata_path),
        "preflight_report_path": path_to_str(report_path),
        "preflight_exit_code": preflight_exit_code if preflight_exit_code is not None else "",
        "preflight_is_valid": preflight_is_valid if preflight_is_valid is not None else "",
        "command": value_or(metadata.get("command"), metrics.get("command"), ""),
    }
    if row:
        record["preflight_summary_raw_row_json"] = json.dumps(row, sort_keys=True)
    return record, errors, warnings


def write_table_package(
    *,
    output_dir: Path,
    preflight_summary: Path,
    preflight_rows: list[dict[str, str]],
    included: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    dataset: str | None,
    backbone: str | None,
    required_execution_env: str,
    required_run_mode: str,
    allow_fake_results: bool,
    command: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    inclusion_rows = [inclusion_registry_row(row) for row in included]
    exclusion_rows = excluded

    outputs["inclusion_registry_csv"] = str(
        safe_write_csv(output_dir / "inclusion_registry.csv", inclusion_rows, INCLUSION_REGISTRY_FIELDS)
    )
    outputs["inclusion_registry_json"] = str(
        safe_write_json(
            output_dir / "inclusion_registry.json",
            {
                "source_preflight_summary": str(preflight_summary),
                "num_included": len(included),
                "num_excluded": len(excluded),
                "policy_note": RAW_PAPER_FLAG_NOTE,
                "included": inclusion_rows,
                "excluded": exclusion_rows,
            },
        )
    )

    main_seed_rows = [project_fields(row, MAIN_SEED_FIELDS) for row in included]
    outputs["main_accuracy_seed_rows_csv"] = str(
        safe_write_csv(output_dir / "main_accuracy_seed_rows.csv", main_seed_rows, MAIN_SEED_FIELDS)
    )
    main_summary = summarize_groups(included, MAIN_SUMMARY_GROUP_FIELDS, MAIN_SUMMARY_FIELDS)
    outputs["main_accuracy_summary_csv"] = str(
        safe_write_csv(output_dir / "main_accuracy_summary.csv", main_summary, MAIN_SUMMARY_FIELDS)
    )

    efficiency_seed_rows = [project_fields(row, EFFICIENCY_SEED_FIELDS) for row in included]
    outputs["efficiency_seed_rows_csv"] = str(
        safe_write_csv(output_dir / "efficiency_seed_rows.csv", efficiency_seed_rows, EFFICIENCY_SEED_FIELDS)
    )
    efficiency_summary = summarize_groups(included, EFFICIENCY_SUMMARY_GROUP_FIELDS, EFFICIENCY_SUMMARY_FIELDS)
    outputs["efficiency_summary_csv"] = str(
        safe_write_csv(output_dir / "efficiency_summary.csv", efficiency_summary, EFFICIENCY_SUMMARY_FIELDS)
    )

    cache_seed_source = [row for row in included if row.get("method") in {"tip_adapter", "proto_adapter", "rs_cpc"}]
    cache_seed_rows = [project_fields(row, CACHE_TRADEOFF_SEED_FIELDS) for row in cache_seed_source]
    outputs["cache_tradeoff_seed_rows_csv"] = str(
        safe_write_csv(output_dir / "cache_tradeoff_seed_rows.csv", cache_seed_rows, CACHE_TRADEOFF_SEED_FIELDS)
    )
    cache_summary = summarize_groups(cache_seed_source, CACHE_TRADEOFF_GROUP_FIELDS, CACHE_TRADEOFF_SUMMARY_FIELDS)
    outputs["cache_tradeoff_summary_csv"] = str(
        safe_write_csv(output_dir / "cache_tradeoff_summary.csv", cache_summary, CACHE_TRADEOFF_SUMMARY_FIELDS)
    )

    ablation_source = [row for row in included if row.get("method") == "rs_cpc"]
    ablation_seed_rows = [project_fields(row, ABLATION_SEED_FIELDS) for row in ablation_source]
    outputs["rs_cpc_ablation_seed_rows_csv"] = str(
        safe_write_csv(
            output_dir / "rs_cpc_m_prototype_init_ablation_seed_rows.csv",
            ablation_seed_rows,
            ABLATION_SEED_FIELDS,
        )
    )
    ablation_summary = summarize_groups(ablation_source, ABLATION_GROUP_FIELDS, ABLATION_SUMMARY_FIELDS)
    outputs["rs_cpc_ablation_summary_csv"] = str(
        safe_write_csv(
            output_dir / "rs_cpc_m_prototype_init_ablation_summary.csv",
            ablation_summary,
            ABLATION_SUMMARY_FIELDS,
        )
    )

    per_class_rows, confusion_rows = extract_per_class_and_confusion(included)
    if per_class_rows:
        outputs["per_class_accuracy_seed_rows_csv"] = str(
            safe_write_csv(output_dir / "per_class_accuracy_seed_rows.csv", per_class_rows, PER_CLASS_FIELDS)
        )
    if confusion_rows:
        outputs["confusion_matrix_seed_rows_csv"] = str(
            safe_write_csv(output_dir / "confusion_matrix_seed_rows.csv", confusion_rows, CONFUSION_FIELDS)
        )
    if not per_class_rows and not confusion_rows:
        note = {
            "status": "not_available",
            "reason": "No per_class_acc or confusion_matrix payload was found in included metrics.json files.",
            "does_not_infer_from_predictions": True,
            "policy_note": "Per-class accuracy and confusion matrix were not fabricated.",
        }
        outputs["per_class_confusion_not_available_json"] = str(
            safe_write_json(output_dir / "per_class_confusion_not_available.json", note)
        )
        outputs["per_class_confusion_not_available_md"] = str(
            write_text_no_overwrite(
                output_dir / "per_class_confusion_not_available.md",
                render_not_available_note(note),
            )
        )

    audit = build_audit_summary(
        output_dir=output_dir,
        preflight_summary=preflight_summary,
        preflight_rows=preflight_rows,
        included=included,
        excluded=excluded,
        dataset=dataset,
        backbone=backbone,
        required_execution_env=required_execution_env,
        required_run_mode=required_run_mode,
        allow_fake_results=allow_fake_results,
        outputs=outputs,
        per_class_available=bool(per_class_rows),
        confusion_available=bool(confusion_rows),
        command=command,
    )
    outputs["audit_summary_json"] = str(safe_write_json(output_dir / "day2_table_audit_summary.json", audit))
    outputs["audit_summary_md"] = str(
        write_text_no_overwrite(output_dir / "day2_table_audit_summary.md", render_audit_markdown(audit))
    )
    return outputs


def build_audit_summary(
    *,
    output_dir: Path,
    preflight_summary: Path,
    preflight_rows: list[dict[str, str]],
    included: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    dataset: str | None,
    backbone: str | None,
    required_execution_env: str,
    required_run_mode: str,
    allow_fake_results: bool,
    outputs: dict[str, str],
    per_class_available: bool,
    confusion_available: bool,
    command: str,
) -> dict[str, Any]:
    exclusion_reason_counts = Counter()
    for row in excluded:
        reasons = [item.strip() for item in str(row.get("exclusion_reasons", "")).split(";") if item.strip()]
        for reason in reasons:
            exclusion_reason_counts[reason] += 1
    raw_flag_counts = {
        "metrics_is_paper_result_true": count_value(included, "raw_metrics_is_paper_result", True),
        "metrics_is_paper_result_false": count_value(included, "raw_metrics_is_paper_result", False),
        "metadata_is_paper_result_true": count_value(included, "raw_metadata_is_paper_result", True),
        "metadata_is_paper_result_false": count_value(included, "raw_metadata_is_paper_result", False),
        "metrics_eligible_for_paper_tables_true": count_value(included, "raw_metrics_eligible_for_paper_tables", True),
        "metrics_eligible_for_paper_tables_false": count_value(included, "raw_metrics_eligible_for_paper_tables", False),
        "metadata_eligible_for_paper_tables_true": count_value(included, "raw_metadata_eligible_for_paper_tables", True),
        "metadata_eligible_for_paper_tables_false": count_value(included, "raw_metadata_eligible_for_paper_tables", False),
    }
    return {
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command,
        "source_script": "scripts/export_verified_tables_from_preflight.py",
        "source_preflight_summary": str(preflight_summary),
        "output_dir": str(output_dir),
        "dataset": dataset,
        "backbone": backbone,
        "required_execution_env": required_execution_env,
        "required_run_mode": required_run_mode,
        "explicitly_excluded_run_modes": sorted(EXCLUDED_RUN_MODES),
        "allow_fake_results": allow_fake_results,
        "does_not_scan_raw_root": True,
        "does_not_run_experiments": True,
        "does_not_modify_results_raw": True,
        "num_preflight_rows": len(preflight_rows),
        "num_included": len(included),
        "num_excluded": len(excluded),
        "included_methods": sorted({str(row.get("method")) for row in included}),
        "included_run_modes": sorted({str(row.get("run_mode")) for row in included}),
        "included_execution_envs": sorted({str(row.get("execution_env")) for row in included}),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "raw_paper_flag_counts": raw_flag_counts,
        "policy_note": RAW_PAPER_FLAG_NOTE,
        "paper_facing_status": "not_marked_as_final_paper_tables",
        "per_class_accuracy_status": "available" if per_class_available else "not_available",
        "confusion_matrix_status": "available" if confusion_available else "not_available",
        "outputs": outputs,
    }


def summarize_groups(
    rows: list[dict[str, Any]],
    group_fields: list[str],
    output_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group_rows = grouped[key]
        summary = {field: value for field, value in zip(group_fields, key)}
        top1_values = numeric_values(group_rows, "top1_acc")
        top1_stats = mean_std(top1_values)
        seeds = sorted({str(row.get("seed", "")) for row in group_rows if str(row.get("seed", "")) != ""}, key=natural_sort_key)
        summary.update(
            {
                "mean_top1_acc": top1_stats["mean"],
                "std_top1_acc": top1_stats["std"],
                "num_seeds": len(seeds),
                "seeds": " ".join(seeds),
                "num_rows": len(group_rows),
                "result_file_paths": " ".join(str(row.get("metrics_json_path", "")) for row in group_rows),
                "run_dirs": " ".join(str(row.get("run_dir", "")) for row in group_rows),
                "preflight_report_paths": " ".join(str(row.get("preflight_report_path", "")) for row in group_rows),
                "cache_entries_values": unique_join(group_rows, "cache_entries"),
                "mean_cache_entries": mean_or_empty(numeric_values(group_rows, "cache_entries")),
                "trainable_params_values": unique_join(group_rows, "trainable_params"),
                "mean_trainable_params": mean_or_empty(numeric_values(group_rows, "trainable_params")),
                "mean_training_time_sec": mean_or_empty(numeric_values(group_rows, "training_time_sec")),
                "mean_inference_time_sec": mean_or_empty(numeric_values(group_rows, "inference_time_sec")),
                "mean_images_per_second": mean_or_empty(numeric_values(group_rows, "images_per_second")),
                "gpu_memory_mb_values": unique_join(group_rows, "gpu_memory_mb"),
                "compression_ratio_values": unique_join(group_rows, "compression_ratio"),
                "policy_note": RAW_PAPER_FLAG_NOTE,
            }
        )
        summaries.append(project_fields(summary, output_fields))
    return summaries


def extract_per_class_and_confusion(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            metrics = read_json(row["metrics_json_path"])
        except Exception:
            continue
        for split_name, payload in per_split_payloads(metrics):
            for class_row in normalize_per_class_accuracy(payload.get("per_class_acc")):
                per_class_rows.append(
                    {
                        **per_class_context(row, split_name),
                        "class_name": class_row.get("class_name", ""),
                        "class_idx": class_row.get("class_idx", ""),
                        "num_samples": class_row.get("num_samples", ""),
                        "num_correct": class_row.get("num_correct", ""),
                        "accuracy": class_row.get("accuracy", ""),
                    }
                )
            matrix = payload.get("confusion_matrix")
            if isinstance(matrix, list) and matrix:
                for true_idx, matrix_row in enumerate(matrix):
                    if not isinstance(matrix_row, list):
                        continue
                    for pred_idx, count in enumerate(matrix_row):
                        confusion_rows.append(
                            {
                                **per_class_context(row, split_name),
                                "true_class_idx": true_idx,
                                "pred_class_idx": pred_idx,
                                "count": count,
                            }
                        )
    return per_class_rows, confusion_rows


def per_split_payloads(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    top_level: dict[str, Any] = {}
    if "per_class_acc" in metrics:
        top_level["per_class_acc"] = metrics.get("per_class_acc")
    if "confusion_matrix" in metrics:
        top_level["confusion_matrix"] = metrics.get("confusion_matrix")
    if top_level:
        payloads.append(("primary", top_level))
    per_split = metrics.get("per_split")
    if isinstance(per_split, dict):
        for split_name, payload in sorted(per_split.items(), key=lambda item: str(item[0])):
            if isinstance(payload, dict):
                payloads.append((str(split_name), payload))
    return payloads


def per_class_context(row: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "shot": row.get("shot", ""),
        "shot_label": row.get("shot_label", ""),
        "backbone": row.get("backbone", ""),
        "method": row.get("method", ""),
        "method_variant": row.get("method_variant", ""),
        "seed": row.get("seed", ""),
        "split": split_name,
        "num_prototypes_per_class": row.get("num_prototypes_per_class", ""),
        "prototype_init": row.get("prototype_init", ""),
        "run_dir": row.get("run_dir", ""),
        "metrics_json_path": row.get("metrics_json_path", ""),
    }


def normalize_per_class_accuracy(per_class_acc: Any) -> list[dict[str, Any]]:
    if isinstance(per_class_acc, list):
        return [row for row in per_class_acc if isinstance(row, dict)]
    if isinstance(per_class_acc, dict):
        rows = []
        for class_name, value in sorted(per_class_acc.items(), key=lambda item: str(item[0])):
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("class_name", class_name)
                rows.append(row)
            else:
                rows.append({"class_name": class_name, "accuracy": value})
        return rows
    return []


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{key: value for key, value in row.items()} for row in reader]


def read_json_for_audit(path: Path | None, label: str, errors: list[str]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        errors.append(f"failed to read {label}: {path}: {exc}")
        return {}


def resolve_path_from_row(row: dict[str, str], keys: list[str], base_dir: Path) -> Path | None:
    value = first_value(row, keys)
    if value is None:
        return None
    return resolve_path(value, base_dir)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def first_value(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def normalize_exit_code(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def extract_top1_acc(metrics: dict[str, Any]) -> tuple[float | None, str]:
    value = float_or_none(metrics.get("top1_acc"))
    if value is not None:
        return value, "top1_acc"
    top1_by_split = metrics.get("top1_acc_by_split")
    if isinstance(top1_by_split, dict):
        if "test" in top1_by_split:
            value = float_or_none(top1_by_split.get("test"))
            if value is not None:
                return value, "top1_acc_by_split.test"
        for split_name, split_value in sorted(top1_by_split.items(), key=lambda item: str(item[0])):
            value = float_or_none(split_value)
            if value is not None:
                return value, f"top1_acc_by_split.{split_name}"
    per_split = metrics.get("per_split")
    if isinstance(per_split, dict):
        if isinstance(per_split.get("test"), dict):
            value = float_or_none(per_split["test"].get("top1_acc"))
            if value is not None:
                return value, "per_split.test.top1_acc"
        for split_name, payload in sorted(per_split.items(), key=lambda item: str(item[0])):
            if isinstance(payload, dict):
                value = float_or_none(payload.get("top1_acc"))
                if value is not None:
                    return value, f"per_split.{split_name}.top1_acc"
    return None, ""


def recover_rs_cpc_m(*, metrics: dict[str, Any], metadata: dict[str, Any], run_dir: Path | None) -> dict[str, Any]:
    candidates: list[tuple[str, int]] = []
    for source, value in [
        ("metrics.num_prototypes_per_class", metrics.get("num_prototypes_per_class")),
        ("metrics.prototypes_per_class", metrics.get("prototypes_per_class")),
        ("metrics.M", metrics.get("M")),
        ("metadata.num_prototypes_per_class", metadata.get("num_prototypes_per_class")),
        ("metadata.prototypes_per_class", metadata.get("prototypes_per_class")),
        ("metadata.M", metadata.get("M")),
    ]:
        parsed = int_or_none(value)
        if parsed is not None:
            candidates.append((source, parsed))
    if run_dir is not None:
        parsed = m_from_run_dir(run_dir)
        if parsed is not None:
            candidates.append(("run_dir.M_x", parsed))
    command = str(value_or(metadata.get("command"), metrics.get("command"), ""))
    parsed = m_from_command(command)
    if parsed is not None:
        candidates.append(("command.--M", parsed))
    parsed = m_from_cache_entries(metrics)
    if parsed is not None:
        candidates.append(("cache_entries/num_classes", parsed))

    values = {value for _, value in candidates}
    if not values:
        return {"value": None, "sources": [], "error": "could not recover RS-CPC M"}
    sources = [f"{source}={value}" for source, value in candidates]
    if len(values) > 1:
        return {"value": None, "sources": sources, "error": f"conflicting RS-CPC M values: {sources}"}
    return {"value": next(iter(values)), "sources": sources, "error": ""}


def recover_prototype_init(*, metrics: dict[str, Any], metadata: dict[str, Any], run_dir: Path | None) -> dict[str, Any]:
    candidates: list[tuple[str, str]] = []
    for source, value in [
        ("metrics.prototype_init", metrics.get("prototype_init")),
        ("metadata.prototype_init", metadata.get("prototype_init")),
    ]:
        if isinstance(value, str) and value.strip():
            candidates.append((source, value.strip()))
    if run_dir is not None:
        value = prototype_init_from_run_dir(run_dir)
        if value:
            candidates.append(("run_dir.prototype_init", value))
    command = str(value_or(metadata.get("command"), metrics.get("command"), ""))
    value = prototype_init_from_command(command)
    if value:
        candidates.append(("command.--prototype-init", value))
    values = {value for _, value in candidates}
    if not values:
        return {"value": None, "sources": [], "error": "could not recover RS-CPC prototype_init"}
    sources = [f"{source}={value}" for source, value in candidates]
    if len(values) > 1:
        return {"value": None, "sources": sources, "error": f"conflicting RS-CPC prototype_init values: {sources}"}
    return {"value": next(iter(values)), "sources": sources, "error": ""}


def m_from_run_dir(run_dir: Path) -> int | None:
    for part in run_dir.parts:
        match = re.fullmatch(r"M_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def prototype_init_from_run_dir(run_dir: Path) -> str | None:
    parts = list(run_dir.parts)
    for index, part in enumerate(parts[:-1]):
        if re.fullmatch(r"M_\d+", part):
            candidate = parts[index + 1]
            if candidate and not candidate.startswith("seed_"):
                return candidate
    return None


def m_from_command(command: str) -> int | None:
    match = re.search(r"(?:^|\s)(?:--M|--num-prototypes-per-class)(?:\s+|=)(\d+)(?:\s|$)", command)
    if match:
        return int(match.group(1))
    return None


def prototype_init_from_command(command: str) -> str | None:
    match = re.search(r"(?:^|\s)--prototype-init(?:\s+|=)([^\s]+)(?:\s|$)", command)
    if match:
        return match.group(1)
    return None


def m_from_cache_entries(metrics: dict[str, Any]) -> int | None:
    cache_entries = int_or_none(metrics.get("cache_entries"))
    if cache_entries is None or cache_entries <= 0:
        return None
    num_classes = int_or_none(metrics.get("num_classes"))
    if num_classes is None:
        dataset = str(metrics.get("dataset", ""))
        num_classes = KNOWN_NUM_CLASSES.get(dataset)
    if num_classes is None or num_classes <= 0 or cache_entries % num_classes != 0:
        return None
    return cache_entries // num_classes


def derive_compression_ratio(metrics: dict[str, Any]) -> Any:
    original = float_or_none(metrics.get("original_cache_entries"))
    compressed = float_or_none(metrics.get("compressed_cache_entries"))
    if original is None:
        shot = float_or_none(metrics.get("shot"))
        num_classes = float_or_none(metrics.get("num_classes"))
        if shot is not None and num_classes is not None:
            original = shot * num_classes
    if compressed is None:
        compressed = float_or_none(metrics.get("cache_entries"))
    if original is None or compressed is None or compressed == 0:
        return ""
    return original / compressed


def make_method_variant(method: str, m_value: Any, prototype_init: Any) -> str:
    if method == "rs_cpc":
        return f"rs_cpc_M{m_value}_{prototype_init}"
    return method


def shot_label(method: str, shot: Any) -> str:
    if str(shot) not in {"", "None", "null"}:
        return str(shot)
    if method in {"zero_shot", "zero_shot_clip"}:
        return "zero_shot"
    return ""


def make_output_dir(*, output_root: Path, output_dir: Path | None, dataset: str | None, backbone: str | None) -> Path:
    if output_dir is not None:
        destination = output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dataset_part = sanitize_path_part(dataset or "verified")
        backbone_part = sanitize_path_part(backbone or "all_backbones")
        destination = output_root / f"day2_{dataset_part}_{backbone_part}_{timestamp}"
    return unique_dir(destination)


def unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise FileExistsError(f"Could not find non-existing output directory for {path}")


def write_text_no_overwrite(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def render_not_available_note(note: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Per-Class Accuracy and Confusion Matrix",
            "",
            f"Status: {note['status']}",
            "",
            note["reason"],
            "",
            "No per-class accuracy or confusion matrix was inferred from predictions or fabricated.",
            "",
        ]
    )


def render_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Day 2 Table Audit",
        "",
        f"Source preflight summary: `{audit['source_preflight_summary']}`",
        f"Output directory: `{audit['output_dir']}`",
        "",
        "This package is a verified analysis table generated from the full-matrix whitelist and post-run preflight.",
        "It preserves the raw `is_paper_result` and `eligible_for_paper_tables` flags and does not mark these rows as final paper-facing results.",
        "",
        "## Counts",
        "",
        f"- Preflight rows: {audit['num_preflight_rows']}",
        f"- Included rows: {audit['num_included']}",
        f"- Excluded rows: {audit['num_excluded']}",
        f"- Included execution envs: {', '.join(audit['included_execution_envs'])}",
        f"- Included run modes: {', '.join(audit['included_run_modes'])}",
        "",
        "## Raw Paper Flags",
        "",
    ]
    for key, value in audit["raw_paper_flag_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Per-Class and Confusion Matrix",
            "",
            f"- Per-class accuracy: {audit['per_class_accuracy_status']}",
            f"- Confusion matrix: {audit['confusion_matrix_status']}",
            "",
            "## Exclusion Reasons",
            "",
        ]
    )
    if audit["exclusion_reason_counts"]:
        for reason, count in audit["exclusion_reason_counts"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Outputs", ""])
    for key, path in audit["outputs"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def inclusion_registry_row(row: dict[str, Any]) -> dict[str, Any]:
    return project_fields(row, INCLUSION_REGISTRY_FIELDS)


def project_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: csv_value(row.get(field, "")) for field in fields}


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def count_value(rows: list[dict[str, Any]], key: str, expected: Any) -> int:
    return sum(1 for row in rows if row.get(key) is expected)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = float_or_none(row.get(key))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": "", "std": ""}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance)}


def mean_or_empty(values: list[float]) -> Any:
    if not values:
        return ""
    return sum(values) / len(values)


def unique_join(rows: list[dict[str, Any]], key: str) -> str:
    values = [str(row.get(key, "")) for row in rows if str(row.get(key, "")) != ""]
    return " ".join(sorted(set(values), key=natural_sort_key))


def natural_sort_key(value: Any) -> list[Any]:
    parts: list[Any] = []
    for part in re.split(r"(\d+)", str(value)):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def value_or(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_or_empty(value: Any) -> Any:
    parsed = int_or_none(value)
    return parsed if parsed is not None else ""


def sanitize_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def path_to_str(path: Path | None) -> str:
    return str(path) if path is not None else ""


INCLUSION_REGISTRY_FIELDS = [
    "preflight_summary_row",
    "include",
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "top1_acc",
    "cache_entries",
    "num_prototypes_per_class",
    "prototype_init",
    "rs_cpc_m_sources",
    "prototype_init_sources",
    "execution_env",
    "run_mode",
    "raw_metrics_is_paper_result",
    "raw_metadata_is_paper_result",
    "raw_metrics_eligible_for_paper_tables",
    "raw_metadata_eligible_for_paper_tables",
    "verified_analysis_table",
    "paper_facing_policy_status",
    "run_dir",
    "metrics_json_path",
    "metadata_json_path",
    "preflight_report_path",
    "preflight_exit_code",
    "preflight_is_valid",
    "warnings",
]

MAIN_SEED_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "top1_acc",
    "top1_acc_source",
    "num_prototypes_per_class",
    "prototype_init",
    "cache_entries",
    "execution_env",
    "run_mode",
    "raw_metrics_is_paper_result",
    "raw_metrics_eligible_for_paper_tables",
    "run_dir",
    "metrics_json_path",
    "metadata_json_path",
    "preflight_report_path",
]
MAIN_SUMMARY_GROUP_FIELDS = ["dataset", "shot_label", "backbone", "method", "method_variant", "num_prototypes_per_class", "prototype_init"]
MAIN_SUMMARY_FIELDS = [
    *MAIN_SUMMARY_GROUP_FIELDS,
    "mean_top1_acc",
    "std_top1_acc",
    "num_seeds",
    "seeds",
    "num_rows",
    "cache_entries_values",
    "result_file_paths",
    "run_dirs",
    "preflight_report_paths",
    "policy_note",
]

EFFICIENCY_SEED_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "cache_entries",
    "trainable_params",
    "training_time_sec",
    "inference_time_sec",
    "images_per_second",
    "gpu_memory_mb",
    "num_prototypes_per_class",
    "prototype_init",
    "run_dir",
    "metrics_json_path",
    "preflight_report_path",
]
EFFICIENCY_SUMMARY_GROUP_FIELDS = MAIN_SUMMARY_GROUP_FIELDS
EFFICIENCY_SUMMARY_FIELDS = [
    *EFFICIENCY_SUMMARY_GROUP_FIELDS,
    "mean_cache_entries",
    "cache_entries_values",
    "mean_trainable_params",
    "trainable_params_values",
    "mean_training_time_sec",
    "mean_inference_time_sec",
    "mean_images_per_second",
    "gpu_memory_mb_values",
    "num_seeds",
    "seeds",
    "num_rows",
    "result_file_paths",
    "run_dirs",
    "preflight_report_paths",
    "policy_note",
]

CACHE_TRADEOFF_SEED_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "num_prototypes_per_class",
    "prototype_init",
    "cache_entries",
    "compression_ratio",
    "top1_acc",
    "inference_time_sec",
    "images_per_second",
    "gpu_memory_mb",
    "run_dir",
    "metrics_json_path",
    "preflight_report_path",
]
CACHE_TRADEOFF_GROUP_FIELDS = MAIN_SUMMARY_GROUP_FIELDS
CACHE_TRADEOFF_SUMMARY_FIELDS = [
    *CACHE_TRADEOFF_GROUP_FIELDS,
    "cache_entries_values",
    "compression_ratio_values",
    "mean_top1_acc",
    "std_top1_acc",
    "num_seeds",
    "seeds",
    "mean_inference_time_sec",
    "mean_images_per_second",
    "gpu_memory_mb_values",
    "num_rows",
    "result_file_paths",
    "run_dirs",
    "preflight_report_paths",
    "policy_note",
]

ABLATION_SEED_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "num_prototypes_per_class",
    "prototype_init",
    "cache_entries",
    "compression_ratio",
    "top1_acc",
    "training_time_sec",
    "inference_time_sec",
    "images_per_second",
    "gpu_memory_mb",
    "rs_cpc_m_sources",
    "prototype_init_sources",
    "run_dir",
    "metrics_json_path",
    "preflight_report_path",
]
ABLATION_GROUP_FIELDS = ["dataset", "shot_label", "backbone", "num_prototypes_per_class", "prototype_init", "method_variant"]
ABLATION_SUMMARY_FIELDS = [
    *ABLATION_GROUP_FIELDS,
    "cache_entries_values",
    "compression_ratio_values",
    "mean_top1_acc",
    "std_top1_acc",
    "num_seeds",
    "seeds",
    "mean_training_time_sec",
    "mean_inference_time_sec",
    "mean_images_per_second",
    "gpu_memory_mb_values",
    "num_rows",
    "result_file_paths",
    "run_dirs",
    "preflight_report_paths",
    "policy_note",
]

PER_CLASS_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "split",
    "num_prototypes_per_class",
    "prototype_init",
    "class_name",
    "class_idx",
    "num_samples",
    "num_correct",
    "accuracy",
    "run_dir",
    "metrics_json_path",
]
CONFUSION_FIELDS = [
    "dataset",
    "shot",
    "shot_label",
    "backbone",
    "method",
    "method_variant",
    "seed",
    "split",
    "num_prototypes_per_class",
    "prototype_init",
    "true_class_idx",
    "pred_class_idx",
    "count",
    "run_dir",
    "metrics_json_path",
]


if __name__ == "__main__":
    main()
