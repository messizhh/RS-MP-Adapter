#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


DEFAULT_INCLUDE_METHODS = ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"]
SUMMARY_FIELDS = [
    "method",
    "shot",
    "M",
    "prototype_init",
    "cache_entries",
    "val_top1_acc",
    "test_top1_acc",
    "best_split_or_primary_metric",
    "run_dir",
    "run_id",
    "run_mode",
    "is_paper_result",
    "eligible_for_paper_tables",
    "result_preflight_status",
    "num_candidate_runs",
]
LOCAL_VALIDATION_RUN_MODE = "local_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cached local_validation result runs without modifying results.")
    parser.add_argument("--results-root", default="results/raw")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/local_validation_summaries")
    parser.add_argument("--include-methods", nargs="+", default=DEFAULT_INCLUDE_METHODS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_local_validation_results(
        results_root=args.results_root,
        dataset=args.dataset,
        backbone=args.backbone,
        seed=args.seed,
        output_dir=args.output_dir,
        include_methods=args.include_methods,
        command=shlex.join(sys.argv),
    )
    print(f"summary_dir={result['summary_dir']}")
    print(f"csv_path={result['csv_path']}")
    print(f"markdown_path={result['markdown_path']}")
    print(f"json_path={result['json_path']}")
    print(f"num_summary_rows={result['num_summary_rows']}")


def summarize_local_validation_results(
    *,
    results_root: str | Path,
    dataset: str,
    backbone: str,
    seed: int,
    output_dir: str | Path,
    include_methods: list[str],
    command: str | None = None,
) -> dict[str, Any]:
    results_path = Path(results_root)
    output_path = Path(output_dir)
    ensure_not_results_raw_output(output_path)

    candidates, excluded_reasons = collect_candidates(
        results_root=results_path,
        dataset=dataset,
        backbone=backbone,
        seed=seed,
        include_methods=include_methods,
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["combo_key"]].append(candidate)

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        selected = max(group, key=latest_sort_key)
        rows.append(make_summary_row(selected, num_candidate_runs=len(group)))
    rows.sort(key=lambda row: summary_sort_key(row, include_methods))

    created_at = utc_now_iso()
    summary_dir = unique_summary_dir(output_path, dataset=dataset, backbone=backbone, seed=seed)
    summary_metadata = {
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
        "created_at": created_at,
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/summarize_local_validation_results.py",
    }
    summary_payload = {
        **summary_metadata,
        "results_root": str(results_path),
        "summary_dir": str(summary_dir),
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "include_methods": include_methods,
        "selection_policy": "latest run per method/shot/M/prototype_init using end_time, start_time, run_id, and path order",
        "filters": {
            "run_mode": LOCAL_VALIDATION_RUN_MODE,
            "is_paper_result": False,
            "eligible_for_paper_tables": False,
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
        },
        "counts": {
            "num_scanned_metrics_json": sum(excluded_reasons.values()) + len(candidates),
            "num_included_candidate_runs": len(candidates),
            "num_summary_rows": len(rows),
            "num_excluded_runs": sum(excluded_reasons.values()),
        },
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
        "rows": rows,
    }

    csv_path = safe_write_csv(summary_dir / "local_validation_summary.csv", rows, SUMMARY_FIELDS)
    markdown_path = write_text_no_overwrite(
        summary_dir / "local_validation_summary.md",
        render_markdown_summary(
            rows=rows,
            dataset=dataset,
            backbone=backbone,
            seed=seed,
            results_root=results_path,
            created_at=created_at,
            excluded_reasons=excluded_reasons,
        ),
    )
    json_path = safe_write_json(summary_dir / "local_validation_summary.json", summary_payload)
    return {
        "summary_dir": str(summary_dir),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "num_summary_rows": len(rows),
        "rows": rows,
    }


def collect_candidates(
    *,
    results_root: Path,
    dataset: str,
    backbone: str,
    seed: int,
    include_methods: list[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    if not results_root.exists():
        return candidates, excluded_reasons

    for metrics_path in sorted(results_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            excluded_reasons["missing_metadata_json"] += 1
            continue
        try:
            metrics = read_json(metrics_path)
            metadata = read_json(metadata_path)
        except Exception:
            excluded_reasons["unreadable_json"] += 1
            continue
        reason = local_validation_exclusion_reason(
            metrics=metrics,
            metadata=metadata,
            dataset=dataset,
            backbone=backbone,
            seed=seed,
            include_methods=include_methods,
        )
        if reason is not None:
            excluded_reasons[reason] += 1
            continue
        method = str(metrics["method"])
        candidates.append(
            {
                "metrics": metrics,
                "metadata": metadata,
                "metrics_path": metrics_path,
                "metadata_path": metadata_path,
                "run_dir": run_dir,
                "combo_key": combo_key(method, metrics, metadata),
            }
        )
    return candidates, excluded_reasons


def local_validation_exclusion_reason(
    *,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    dataset: str,
    backbone: str,
    seed: int,
    include_methods: list[str],
) -> str | None:
    for field in ["method", "dataset", "backbone", "seed", "run_mode", "is_paper_result", "eligible_for_paper_tables"]:
        if field not in metrics or field not in metadata:
            return f"missing_{field}"
    if metrics.get("method") != metadata.get("method"):
        return "method_mismatch_between_metrics_and_metadata"
    method = str(metrics.get("method"))
    if method not in include_methods:
        return "method_not_included"
    if metrics.get("dataset") != dataset or metadata.get("dataset") != dataset:
        return "dataset_mismatch"
    if metrics.get("backbone") != backbone or metadata.get("backbone") != backbone:
        return "backbone_mismatch"
    if int_or_none(metrics.get("seed")) != seed or int_or_none(metadata.get("seed")) != seed:
        return "seed_mismatch"
    if metrics.get("run_mode") != LOCAL_VALIDATION_RUN_MODE or metadata.get("run_mode") != LOCAL_VALIDATION_RUN_MODE:
        return "not_local_validation"
    if metrics.get("is_paper_result") is not False or metadata.get("is_paper_result") is not False:
        return "is_paper_result_not_false"
    if metrics.get("eligible_for_paper_tables") is not False or metadata.get("eligible_for_paper_tables") is not False:
        return "eligible_for_paper_tables_not_false"
    return None


def combo_key(method: str, metrics: dict[str, Any], metadata: dict[str, Any]) -> tuple[Any, ...]:
    if method == "zero_shot":
        return (method, None, None, "")
    shot = first_present(metrics, metadata, "shot")
    if method == "rs_cpc":
        return (method, int_or_none(shot), int_or_none(first_present(metrics, metadata, "M", "num_prototypes_per_class", "prototypes_per_class")), str(first_present(metrics, metadata, "prototype_init") or ""))
    return (method, int_or_none(shot), None, "")


def latest_sort_key(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    metrics = candidate["metrics"]
    metadata = candidate["metadata"]
    return (
        str(first_present(metadata, metrics, "end_time") or ""),
        str(first_present(metadata, metrics, "start_time") or ""),
        str(first_present(metadata, metrics, "run_id") or candidate["run_dir"].name),
        str(candidate["metrics_path"]),
    )


def make_summary_row(candidate: dict[str, Any], *, num_candidate_runs: int) -> dict[str, Any]:
    metrics = candidate["metrics"]
    metadata = candidate["metadata"]
    method = str(metrics.get("method", ""))
    val_top1 = split_top1(metrics, "val")
    test_top1 = split_top1(metrics, "test")
    return {
        "method": method,
        "shot": "" if method == "zero_shot" else empty_if_none(first_present(metrics, metadata, "shot")),
        "M": empty_if_none(first_present(metrics, metadata, "M", "num_prototypes_per_class", "prototypes_per_class")),
        "prototype_init": "" if method != "rs_cpc" else str(first_present(metrics, metadata, "prototype_init") or ""),
        "cache_entries": empty_if_none(first_present(metrics, metadata, "cache_entries")),
        "val_top1_acc": empty_if_none(val_top1),
        "test_top1_acc": empty_if_none(test_top1),
        "best_split_or_primary_metric": primary_metric_name(metrics, val_top1=val_top1, test_top1=test_top1),
        "run_dir": str(candidate["run_dir"]),
        "run_id": str(first_present(metrics, metadata, "run_id") or candidate["run_dir"].name),
        "run_mode": str(first_present(metrics, metadata, "run_mode") or ""),
        "is_paper_result": bool(first_present(metrics, metadata, "is_paper_result")),
        "eligible_for_paper_tables": bool(first_present(metrics, metadata, "eligible_for_paper_tables")),
        "result_preflight_status": discover_result_preflight_status(candidate),
        "num_candidate_runs": num_candidate_runs,
    }


def split_top1(metrics: dict[str, Any], split: str) -> Any:
    top1_by_split = metrics.get("top1_acc_by_split")
    if isinstance(top1_by_split, dict) and split in top1_by_split:
        return top1_by_split[split]
    per_split = metrics.get("per_split")
    if isinstance(per_split, dict):
        split_payload = per_split.get(split)
        if isinstance(split_payload, dict) and "top1_acc" in split_payload:
            return split_payload["top1_acc"]
    return None


def primary_metric_name(metrics: dict[str, Any], *, val_top1: Any, test_top1: Any) -> str:
    if test_top1 is not None:
        return "test_top1_acc"
    if val_top1 is not None:
        return "val_top1_acc"
    if metrics.get("top1_acc") is not None:
        return "top1_acc"
    return ""


def discover_result_preflight_status(candidate: dict[str, Any]) -> str:
    metrics = candidate["metrics"]
    metadata = candidate["metadata"]
    run_dir = candidate["run_dir"]
    for field in ["result_preflight_status", "result_run_preflight_status"]:
        value = first_present(metrics, metadata, field)
        if value is not None:
            return str(value)
    for field in ["result_preflight_report_path", "result_run_preflight_report_path", "result_preflight_report"]:
        value = first_present(metrics, metadata, field)
        if value:
            status = status_from_preflight_report(resolve_path(value, run_dir))
            if status:
                return status
    for path in [run_dir / "result_run_preflight_report.json", run_dir / "preflight" / "result_run_preflight_report.json"]:
        status = status_from_preflight_report(path)
        if status:
            return status
    return "not_discovered"


def status_from_preflight_report(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        report = read_json(path)
    except Exception:
        return "unreadable"
    if "is_valid" in report:
        return f"valid={str(bool(report['is_valid'])).lower()}"
    return "discovered"


def resolve_path(value: Any, run_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    return path


def render_markdown_summary(
    *,
    rows: list[dict[str, Any]],
    dataset: str,
    backbone: str,
    seed: int,
    results_root: Path,
    created_at: str,
    excluded_reasons: Counter[str],
) -> str:
    lines = [
        "# Local Validation Summary",
        "",
        "This is local_validation only.",
        "Not eligible for paper tables.",
        "Do not cite as final result.",
        "",
        "This summary reads existing `metrics.json` and `metadata.json` files only. It does not compute logits, recompute accuracy, evaluate a model, train a model, modify results, delete results, or write `results/raw`.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Backbone: `{backbone}`",
        f"- Seed: `{seed}`",
        f"- Results root: `{results_root}`",
        f"- Created at: `{created_at}`",
        f"- Summary rows: `{len(rows)}`",
        "",
    ]
    if excluded_reasons:
        lines.extend(["Excluded run counts:", ""])
        for reason, count in sorted(excluded_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.extend(markdown_table(rows))
    lines.append("")
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No matching local_validation runs were found."]
    lines = [
        "| Method | Shot | M | Prototype Init | Cache Entries | Val Top-1 | Test Top-1 | Run ID | Candidates |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {shot} | {M} | {prototype_init} | {cache_entries} | {val_top1_acc} | {test_top1_acc} | {run_id} | {num_candidate_runs} |".format(
                **{key: markdown_cell(row.get(key, "")) for key in SUMMARY_FIELDS}
            )
        )
    return lines


def markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def summary_sort_key(row: dict[str, Any], include_methods: list[str]) -> tuple[int, int, int, str]:
    method_order = {method: index for index, method in enumerate(include_methods)}
    return (
        method_order.get(str(row.get("method", "")), len(method_order)),
        int_or_none(row.get("shot")) if int_or_none(row.get("shot")) is not None else -1,
        int_or_none(row.get("M")) if int_or_none(row.get("M")) is not None else -1,
        str(row.get("prototype_init", "")),
    )


def unique_summary_dir(output_dir: Path, *, dataset: str, backbone: str, seed: int) -> Path:
    base_dir = output_dir / f"{dataset}_{backbone}_seed{seed}"
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique local validation summary directory under {base_dir}")


def ensure_not_results_raw_output(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("local validation summaries must not be written under results/raw")


def write_text_no_overwrite(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8")
    return path


def first_present(*payloads_and_fields: Any) -> Any:
    payloads: list[dict[str, Any]] = []
    fields: list[Any] = []
    for item in payloads_and_fields:
        if isinstance(item, dict) and not fields:
            payloads.append(item)
        else:
            fields.append(item)
    for payload in payloads:
        for field in fields:
            if field in payload and payload[field] is not None:
                return payload[field]
    return None


def empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    main()
