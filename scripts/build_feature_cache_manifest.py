#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
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
    "seed",
    "shot",
    "split",
    "split_id",
    "split_name",
    "split_file_stem",
    "base_split",
    "split_path",
    "split_section",
    "image_count",
    "num_samples",
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
SPLIT_SECTION_NAMES = {"train", "val", "test", "support"}
SHOT_SPLIT_RE = re.compile(r"shot_(\d+)_seed(\d+)")
BASE_SPLIT_RE = re.compile(r"base(?:_split)?_seed(\d+)")


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
    raw_entries = [entry_from_summary(path) for path in summary_paths]
    entries, ignored_stale_entries = deduplicate_manifest_entries(raw_entries)
    summary = summarize_manifest_entries(
        entries,
        ignored_stale_entries=ignored_stale_entries,
        execution_env=execution_env,
        run_mode=run_mode,
        features_root=root,
    )

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
    if entry.get("num_samples") is None:
        entry["num_samples"] = summary.get("image_count")
    return {field: entry.get(field) for field in MANIFEST_FIELDS}


def deduplicate_manifest_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(manifest_logical_key(entry), []).append(entry)

    kept_entries: list[dict[str, Any]] = []
    ignored_entries: list[dict[str, Any]] = []
    for logical_key, group in grouped.items():
        selected = select_preferred_manifest_entry(group)
        kept_entries.append(selected)
        if len(group) <= 1:
            continue
        for entry in group:
            if entry is selected:
                continue
            ignored_entries.append(stale_entry_record(entry, selected, logical_key))

    return (
        sorted(kept_entries, key=lambda entry: str(entry.get("summary_path") or "")),
        sorted(ignored_entries, key=lambda entry: str(entry.get("summary_path") or "")),
    )


def select_preferred_manifest_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise ValueError("cannot select a preferred manifest entry from an empty list")
    return max(entries, key=entry_priority_key)


def entry_priority_key(entry: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        int(is_valid_feature_summary(entry)),
        int(has_explicit_split_identity(entry)),
        metadata_completeness_score(entry),
        newest_timestamp(entry),
        str(entry.get("summary_path") or ""),
    )


def manifest_logical_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = canonical_split_identity(entry)
    if identity is None:
        identity = f"summary:{entry.get('summary_path') or id(entry)}"
    return (
        normalized_text(entry.get("dataset"), "__missing_dataset__"),
        normalized_text(entry.get("backbone"), "__missing_backbone__"),
        canonical_split_section(entry),
        identity,
    )


def canonical_split_section(entry: dict[str, Any]) -> str:
    section = entry.get("split_section")
    if isinstance(section, str) and section in SPLIT_SECTION_NAMES:
        return section
    for key in ("summary_path", "run_dir", "feature_cache_path"):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        for part in path.parts:
            if part in SPLIT_SECTION_NAMES:
                return part
    return normalized_text(section, "__missing_section__")


def canonical_split_identity(entry: dict[str, Any]) -> str | None:
    for key in ("base_split", "split_id", "split_name", "split_file_stem"):
        identity = split_identity_from_value(entry.get(key), allow_plain=True)
        if identity:
            return identity
    for key in ("split", "split_path", "summary_path", "run_dir", "feature_cache_path"):
        identity = split_identity_from_value(entry.get(key), allow_plain=False)
        if identity:
            return identity
    return None


def has_explicit_split_identity(entry: dict[str, Any]) -> bool:
    return any(
        split_identity_from_value(entry.get(key), allow_plain=True)
        for key in ("base_split", "split_id", "split_name", "split_file_stem")
    )


def split_identity_from_value(value: Any, *, allow_plain: bool) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    path = Path(text)
    candidates = [text, path.name, path.stem, *path.parts]
    for candidate in candidates:
        shot_match = SHOT_SPLIT_RE.search(candidate)
        if shot_match:
            return f"shot_{shot_match.group(1)}_seed{shot_match.group(2)}"
        base_match = BASE_SPLIT_RE.search(candidate)
        if base_match:
            return f"base_seed{base_match.group(1)}"
    if not allow_plain:
        return None
    token = path.stem if path.suffix or "/" in text or "\\" in text else text
    token = token.strip()
    if not token or token in SPLIT_SECTION_NAMES:
        return None
    return token


def metadata_completeness_score(entry: dict[str, Any]) -> int:
    fields = [
        "dataset",
        "backbone",
        "seed",
        "shot",
        "split",
        "split_id",
        "split_name",
        "split_file_stem",
        "base_split",
        "split_path",
        "split_section",
        "image_count",
        "num_samples",
        "feature_shape",
        "feature_cache_path",
        "run_dir",
        "checkpoint_loaded",
        "final_weights_loaded_from_checkpoint",
        "missing_keys_count",
        "unexpected_keys_count",
    ]
    return sum(1 for field in fields if has_value(entry.get(field)))


def is_valid_feature_summary(entry: dict[str, Any]) -> bool:
    image_count = int_or_none(entry.get("image_count"))
    shape = feature_shape(entry.get("feature_shape"))
    if image_count is None or image_count < 0:
        return False
    if shape and (len(shape) != 2 or shape[0] != image_count):
        return False
    return True


def newest_timestamp(entry: dict[str, Any]) -> str:
    values = [
        str(entry.get(field))
        for field in ("end_time", "start_time", "created_at")
        if isinstance(entry.get(field), str) and entry.get(field)
    ]
    return max(values) if values else ""


def stale_entry_record(
    entry: dict[str, Any], selected: dict[str, Any], logical_key: tuple[str, str, str, str]
) -> dict[str, Any]:
    reason = "duplicate_logical_key_lower_priority"
    if not has_explicit_split_identity(entry) and has_explicit_split_identity(selected):
        reason = "duplicate_without_explicit_split_metadata"
    return {
        "summary_path": entry.get("summary_path"),
        "selected_summary_path": selected.get("summary_path"),
        "dataset": entry.get("dataset"),
        "backbone": entry.get("backbone"),
        "split_section": entry.get("split_section"),
        "split_identity": logical_key[3],
        "logical_key": list(logical_key),
        "image_count": entry.get("image_count"),
        "selected_image_count": selected.get("image_count"),
        "reason": reason,
    }


def normalized_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value:
        return value
    return fallback


def has_value(value: Any) -> bool:
    return value is not None and value != ""


def feature_shape(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if isinstance(item, (int, float))]
    return []


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def summarize_manifest_entries(
    entries: list[dict[str, Any]],
    *,
    ignored_stale_entries: list[dict[str, Any]],
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
        "deduplication_enabled": True,
        "num_ignored_stale_entries": len(ignored_stale_entries),
        "ignored_stale_entries": ignored_stale_entries,
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
