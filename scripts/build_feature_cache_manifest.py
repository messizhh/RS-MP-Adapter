#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


MANIFEST_FIELDS = [
    "summary_path",
    "dataset",
    "backbone",
    "split_path",
    "split_section",
    "image_count",
    "feature_shape",
    "feature_cache_path",
    "run_dir",
    "git_commit",
    "checkpoint_loaded",
    "checkpoint_load_mode",
    "checkpoint_num_tensors",
    "missing_keys_count",
    "unexpected_keys_count",
    "final_weights_loaded_from_checkpoint",
    "final_weight_source",
    "final_checkpoint_load_status",
    "is_real_feature_extraction",
    "is_full_feature_extraction",
    "is_limited_real_extraction",
    "is_paper_result",
    "is_paper_result_candidate",
    "eligible_for_paper_tables",
    "trains_model",
    "evaluates_model",
    "extracts_text_features",
    "saves_predictions",
    "saves_logits",
    "start_time",
    "end_time",
    "total_time_sec",
]

WARNING_FLAGS = [
    "trains_model",
    "evaluates_model",
    "saves_predictions",
    "is_paper_result",
    "eligible_for_paper_tables",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only manifest from feature_extraction_summary.json files.")
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_feature_cache_manifest(
        features_root=args.features_root,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        overwrite=args.overwrite,
    )
    print(f"manifest_json_path={result['manifest_json_path']}")
    print(f"manifest_csv_path={result['manifest_csv_path']}")
    print(f"manifest_summary_path={result['manifest_summary_path']}")


def build_feature_cache_manifest(
    *,
    features_root: str | Path,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = Path(features_root)
    if not root.exists():
        raise FileNotFoundError(f"features root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"features root is not a directory: {root}")

    summary_paths = sorted(root.rglob("feature_extraction_summary.json"))
    entries = [entry_from_summary(path) for path in summary_paths]
    summary = summarize_manifest_entries(entries, execution_env=execution_env, run_mode=run_mode, features_root=root)

    destination = Path(output_dir)
    manifest_json_path = safe_write_json(destination / "feature_cache_manifest.json", {"entries": entries}, overwrite=overwrite)
    manifest_csv_path = safe_write_csv(
        destination / "feature_cache_manifest.csv",
        entries,
        fieldnames=MANIFEST_FIELDS,
        overwrite=overwrite,
    )
    manifest_summary_path = safe_write_json(
        destination / "feature_cache_manifest_summary.json",
        summary,
        overwrite=overwrite,
    )
    return {
        "manifest_json_path": manifest_json_path,
        "manifest_csv_path": manifest_csv_path,
        "manifest_summary_path": manifest_summary_path,
    }


def entry_from_summary(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    entry = {field: summary.get(field) for field in MANIFEST_FIELDS if field != "summary_path"}
    entry["summary_path"] = str(summary_path)
    if entry.get("run_dir") is None:
        entry["run_dir"] = str(summary_path.parent)
    return {field: entry.get(field) for field in MANIFEST_FIELDS}


def summarize_manifest_entries(
    entries: list[dict[str, Any]],
    *,
    execution_env: str,
    run_mode: str,
    features_root: Path,
) -> dict[str, Any]:
    warning_entries = []
    for entry in entries:
        triggered = [flag for flag in WARNING_FLAGS if bool(entry.get(flag))]
        if triggered:
            warning_entries.append(
                {
                    "summary_path": entry.get("summary_path"),
                    "feature_cache_path": entry.get("feature_cache_path"),
                    "dataset": entry.get("dataset"),
                    "backbone": entry.get("backbone"),
                    "split_section": entry.get("split_section"),
                    "warning_flags": triggered,
                }
            )

    return {
        "num_entries": len(entries),
        "datasets": sorted({str(entry.get("dataset")) for entry in entries if entry.get("dataset") is not None}),
        "backbones": sorted({str(entry.get("backbone")) for entry in entries if entry.get("backbone") is not None}),
        "split_sections": sorted(
            {str(entry.get("split_section")) for entry in entries if entry.get("split_section") is not None}
        ),
        "total_images": sum_int(entry.get("image_count") for entry in entries),
        "num_paper_results": count_true(entries, "is_paper_result"),
        "num_eligible_for_paper_tables": count_true(entries, "eligible_for_paper_tables"),
        "num_with_checkpoint_loaded_false": sum(1 for entry in entries if entry.get("checkpoint_loaded") is False),
        "num_with_training_true": count_true(entries, "trains_model"),
        "num_with_evaluation_true": count_true(entries, "evaluates_model"),
        "num_with_predictions_true": count_true(entries, "saves_predictions"),
        "num_with_text_features_true": count_true(entries, "extracts_text_features"),
        "manifest_is_paper_result": False,
        "warning_entries": warning_entries,
        "features_root": str(features_root),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "created_at": utc_now_iso(),
        "source_script": "scripts/build_feature_cache_manifest.py",
        "reads_feature_extraction_summary_only": True,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "saves_predictions": False,
        "saves_logits": False,
    }


def count_true(entries: list[dict[str, Any]], key: str) -> int:
    return sum(1 for entry in entries if bool(entry.get(key)))


def sum_int(values: Any) -> int:
    total = 0
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
        elif isinstance(value, float):
            total += int(value)
    return total


if __name__ == "__main__":
    main()
