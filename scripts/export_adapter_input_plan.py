#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


PLAN_FIELDS = [
    "dataset",
    "backbone",
    "seed",
    "shot_split",
    "shot",
    "method",
    "num_classes",
    "feature_dim",
    "support_entries",
    "candidate_M",
    "is_ready",
    "skip_reason",
    "expected_cache_entries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a read-only adapter input plan from a preflight report.")
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--output-dir", default="outputs/preflight/adapter_input_plans")
    parser.add_argument("--execution-env", default=None)
    parser.add_argument("--run-mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_adapter_input_plan(
        preflight_report_path=args.preflight_report,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"plan_json_path={result['plan_json_path']}")
    print(f"plan_csv_path={result['plan_csv_path']}")


def export_adapter_input_plan(
    *,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    execution_env: str | None = None,
    run_mode: str | None = None,
    command: str | None = None,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    report_path = Path(preflight_report_path)
    report = read_json(report_path)
    rows = build_plan_rows(report)

    dataset = str(report.get("dataset", "unknown_dataset"))
    backbone = str(report.get("backbone", "unknown_backbone"))
    seed = infer_seed(report)
    effective_execution_env = execution_env or str(report.get("execution_env", "unknown"))
    effective_run_mode = run_mode or str(report.get("run_mode", "unknown"))
    plan_dir = unique_dir(output_root / f"{dataset}_{backbone}_{seed}")

    plan = {
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "source_preflight_report": str(report_path),
        "source_preflight_is_valid": bool(report.get("is_valid", False)),
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "num_classes": report.get("num_classes"),
        "feature_dim": report.get("feature_dim"),
        "execution_env": effective_execution_env,
        "run_mode": effective_run_mode,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/export_adapter_input_plan.py",
        "trains_model": False,
        "evaluates_model": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "num_rows": len(rows),
        "num_ready_rows": sum(1 for row in rows if row["is_ready"] is True),
        "num_not_ready_rows": sum(1 for row in rows if row["is_ready"] is False),
        "rows": rows,
    }

    plan_json_path = safe_write_json(plan_dir / "adapter_input_plan.json", plan)
    plan_csv_path = safe_write_csv(plan_dir / "adapter_input_plan.csv", rows, PLAN_FIELDS)
    return {"plan_json_path": plan_json_path, "plan_csv_path": plan_csv_path, "plan_dir": plan_dir}


def build_plan_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    method_summary = report.get("per_method_input_summary")
    if not isinstance(method_summary, dict):
        raise ValueError("preflight report must contain per_method_input_summary")

    rows: list[dict[str, Any]] = []
    base_context = {
        "dataset": report.get("dataset"),
        "backbone": report.get("backbone"),
        "seed": infer_seed(report),
        "num_classes": report.get("num_classes"),
        "feature_dim": report.get("feature_dim"),
    }
    for method, summary in sorted(method_summary.items(), key=lambda item: str(item[0])):
        if method == "rs_cpc":
            rows.extend(build_rs_cpc_rows(base_context, summary))
        elif method in {"tip_adapter", "proto_adapter"}:
            rows.extend(build_single_cache_rows(base_context, method, summary))
    return rows


def build_single_cache_rows(base_context: dict[str, Any], method: str, summary: Any) -> list[dict[str, Any]]:
    per_shot = summary.get("per_shot") if isinstance(summary, dict) else None
    if not isinstance(per_shot, dict):
        return []
    rows = []
    for shot_split, shot_summary in sorted(per_shot.items(), key=lambda item: shot_sort_key(item[0], item[1])):
        if not isinstance(shot_summary, dict):
            continue
        is_ready = bool(shot_summary.get("method_input_ready", False))
        rows.append(
            {
                **base_context,
                "shot_split": shot_split,
                "shot": shot_summary.get("shot"),
                "method": method,
                "support_entries": shot_summary.get("actual_support_entries"),
                "candidate_M": None,
                "is_ready": is_ready,
                "skip_reason": "" if is_ready else "preflight_method_input_not_ready",
                "expected_cache_entries": shot_summary.get("expected_cache_entries"),
            }
        )
    return rows


def build_rs_cpc_rows(base_context: dict[str, Any], summary: Any) -> list[dict[str, Any]]:
    per_shot = summary.get("per_shot") if isinstance(summary, dict) else None
    if not isinstance(per_shot, dict):
        return []
    rows = []
    for shot_split, shot_summary in sorted(per_shot.items(), key=lambda item: shot_sort_key(item[0], item[1])):
        if not isinstance(shot_summary, dict):
            continue
        ready_by_m = shot_summary.get("method_input_ready_by_M")
        expected_by_m = shot_summary.get("expected_cache_entries_by_M")
        if not isinstance(ready_by_m, dict):
            continue
        if not isinstance(expected_by_m, dict):
            expected_by_m = {}
        min_support = int_or_none(shot_summary.get("min_support_per_class"))
        for candidate_m in sorted((int(key) for key in ready_by_m.keys()), key=int):
            is_ready = bool(ready_by_m.get(str(candidate_m), False))
            rows.append(
                {
                    **base_context,
                    "shot_split": shot_split,
                    "shot": shot_summary.get("shot"),
                    "method": "rs_cpc",
                    "support_entries": shot_summary.get("actual_support_entries"),
                    "candidate_M": candidate_m,
                    "is_ready": is_ready,
                    "skip_reason": rs_cpc_skip_reason(is_ready, candidate_m, min_support),
                    "expected_cache_entries": expected_by_m.get(str(candidate_m)),
                }
            )
    return rows


def rs_cpc_skip_reason(is_ready: bool, candidate_m: int, min_support: int | None) -> str:
    if is_ready:
        return ""
    if min_support is not None and candidate_m > min_support:
        return "candidate_M_exceeds_min_support_per_class"
    return "preflight_method_input_not_ready"


def infer_seed(report: dict[str, Any]) -> str:
    base_split = report.get("checked_base_split")
    if isinstance(base_split, dict) and base_split.get("seed") is not None:
        return f"seed{base_split['seed']}"
    checked_shot_splits = report.get("checked_shot_splits")
    if not isinstance(checked_shot_splits, list):
        checked_shot_splits = []
    for value in [
        report.get("manifest_path"),
        report.get("source_preflight_report"),
        *(split.get("input") for split in checked_shot_splits if isinstance(split, dict)),
    ]:
        if isinstance(value, str):
            match = re.search(r"seed(\d+)", value)
            if match:
                return f"seed{match.group(1)}"
    return "seed_unknown"


def shot_sort_key(shot_split: str, shot_summary: Any) -> tuple[int, str]:
    if isinstance(shot_summary, dict) and isinstance(shot_summary.get("shot"), int):
        return int(shot_summary["shot"]), shot_split
    match = re.search(r"shot_(\d+)", shot_split)
    if match:
        return int(match.group(1)), shot_split
    return 10**9, shot_split


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("adapter input plans must not be written under results/raw")


def unique_dir(base_dir: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique plan directory under {base_dir}")


if __name__ == "__main__":
    main()
