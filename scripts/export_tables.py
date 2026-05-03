#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_json, safe_write_csv, safe_write_json


DEFAULT_INCLUDE_RUN_MODES = ["server_full", "server_ablation", "server_benchmark"]
DEFAULT_EXCLUDE_RUN_MODES = ["dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export result tables from metrics JSON files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tables", nargs="+", default=["main", "efficiency"])
    parser.add_argument("--include-run-modes", nargs="+", default=DEFAULT_INCLUDE_RUN_MODES)
    parser.add_argument("--exclude-run-modes", nargs="+", default=DEFAULT_EXCLUDE_RUN_MODES)
    parser.add_argument("--allow-local-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = load_result_jsons(Path(args.input_dir))
    eligible = [
        result
        for result in all_results
        if is_eligible_result(
            result,
            include_run_modes=args.include_run_modes,
            exclude_run_modes=args.exclude_run_modes,
            allow_local_results=args.allow_local_results,
        )
    ]

    outputs: dict[str, str] = {}
    if "main" in args.tables or "main_accuracy" in args.tables:
        outputs["main_accuracy"] = str(export_main_accuracy(eligible, output_dir / "main_accuracy.csv"))
    if "efficiency" in args.tables:
        outputs["efficiency"] = str(export_efficiency(eligible, output_dir / "efficiency.csv"))

    summary = {
        "input_dir": str(Path(args.input_dir)),
        "output_dir": str(output_dir),
        "num_result_json": len(all_results),
        "num_eligible_results": len(eligible),
        "include_run_modes": args.include_run_modes,
        "exclude_run_modes": args.exclude_run_modes,
        "allow_local_results": args.allow_local_results,
        "outputs": outputs,
        "message": "No eligible results found; empty CSV files contain headers only." if not eligible else "Exported eligible result rows.",
    }
    summary_path = safe_write_json(unique_path(output_dir / "summary.json"), summary)
    print(f"summary_path={summary_path}")
    for key, path in outputs.items():
        print(f"{key}_path={path}")


def load_result_jsons(input_dir: Path) -> list[dict[str, Any]]:
    if not input_dir.exists():
        return []
    results = []
    for path in sorted(input_dir.rglob("metrics.json")):
        try:
            result = read_json(path)
        except Exception:
            continue
        result["_metrics_json_path"] = str(path)
        results.append(result)
    return results


def is_eligible_result(
    result: dict[str, Any],
    include_run_modes: list[str],
    exclude_run_modes: list[str],
    allow_local_results: bool,
) -> bool:
    run_mode = str(result.get("run_mode", ""))
    if run_mode in exclude_run_modes:
        return False
    if include_run_modes and run_mode not in include_run_modes:
        return False
    if not allow_local_results and str(result.get("execution_env", "")) == "local_wsl":
        return False
    if result.get("is_paper_result") is False and not allow_local_results:
        return False
    if result.get("fake_or_dry_run") or result.get("uses_fake_data") or result.get("uses_fake_features"):
        return False
    return True


def export_main_accuracy(results: list[dict[str, Any]], path: Path) -> Path:
    rows = [
        {
            "dataset": result.get("dataset", ""),
            "shot": result.get("shot", ""),
            "backbone": result.get("backbone", ""),
            "method": result.get("method", ""),
            "seed": result.get("seed", ""),
            "top1_acc": result.get("top1_acc", ""),
            "run_mode": result.get("run_mode", ""),
            "execution_env": result.get("execution_env", ""),
            "is_paper_result": result.get("is_paper_result", ""),
            "result_json_path": result.get("_metrics_json_path", result.get("result_json_path", "")),
        }
        for result in results
    ]
    return safe_write_csv(unique_path(path), rows, ["dataset", "shot", "backbone", "method", "seed", "top1_acc", "run_mode", "execution_env", "is_paper_result", "result_json_path"])


def export_efficiency(results: list[dict[str, Any]], path: Path) -> Path:
    rows = [
        {
            "dataset": result.get("dataset", ""),
            "shot": result.get("shot", ""),
            "backbone": result.get("backbone", ""),
            "method": result.get("method", ""),
            "cache_entries": result.get("cache_entries", ""),
            "trainable_params": result.get("trainable_params", ""),
            "training_time_sec": result.get("training_time_sec", ""),
            "inference_time_sec": result.get("inference_time_sec", ""),
            "images_per_second": result.get("images_per_second", ""),
            "gpu_memory_mb": result.get("gpu_memory_mb", ""),
            "result_json_path": result.get("_metrics_json_path", result.get("result_json_path", "")),
        }
        for result in results
    ]
    return safe_write_csv(unique_path(path), rows, ["dataset", "shot", "backbone", "method", "cache_entries", "trainable_params", "training_time_sec", "inference_time_sec", "images_per_second", "gpu_memory_mb", "result_json_path"])


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find non-existing output path for {path}")


if __name__ == "__main__":
    main()
