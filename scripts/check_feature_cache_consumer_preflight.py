#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_cache import shape_of_2d
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


BASE_SECTIONS = ["train", "val", "test"]
CONSUMER_FORBIDDEN_TRUE_FLAGS = [
    "is_paper_result",
    "eligible_for_paper_tables",
    "trains_model",
    "evaluates_model",
    "saves_predictions",
    "saves_logits",
    "extracts_text_features",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only feature cache consumer preflight.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--base-split", required=True)
    parser.add_argument("--shot-splits", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_feature_cache_consumer_preflight(
        manifest_path=args.manifest,
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split,
        shot_splits=args.shot_splits,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"feature_cache_consumer_preflight_report_path={report_path}")
    if not is_valid:
        raise SystemExit(1)


def run_feature_cache_consumer_preflight(
    *,
    manifest_path: str | Path,
    dataset: str,
    backbone: str,
    base_split: str,
    shot_splits: list[str],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
) -> tuple[Path, bool]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_source = Path(manifest_path)
    manifest = read_json(manifest_source)
    entries = manifest_entries(manifest)
    resolved_entries = []
    for entry in entries:
        try:
            resolved_entries.append(resolve_entry_from_summary(entry, manifest_source.parent))
        except Exception as exc:
            errors.append(f"failed to read feature extraction summary for manifest entry: {exc}")

    selected_entries = [
        entry for entry in resolved_entries if entry.get("dataset") == dataset and entry.get("backbone") == backbone
    ]
    entries_by_split = {
        split_id: [entry for entry in selected_entries if entry_matches_split(entry, split_id)]
        for split_id in [base_split, *shot_splits]
    }

    found_base_sections = {
        str(entry.get("split_section"))
        for entry in entries_by_split.get(base_split, [])
        if entry.get("split_section") in BASE_SECTIONS
    }
    required_base_sections_found = [section for section in BASE_SECTIONS if section in found_base_sections]
    required_support_splits_found = [
        split_id
        for split_id in shot_splits
        if any(entry.get("split_section") == "support" for entry in entries_by_split.get(split_id, []))
    ]

    for section in BASE_SECTIONS:
        if section not in required_base_sections_found:
            errors.append(f"{base_split} is missing required {section} feature cache summary")
    for split_id in shot_splits:
        if split_id not in required_support_splits_found:
            errors.append(f"{split_id} is missing required support feature cache summary")

    required_entries = [
        entry
        for entry in entries_by_split.get(base_split, [])
        if entry.get("split_section") in BASE_SECTIONS
    ]
    for split_id in shot_splits:
        required_entries.extend(
            entry for entry in entries_by_split.get(split_id, []) if entry.get("split_section") == "support"
        )

    cache_inspections: dict[str, dict[str, Any]] = {}
    for entry in required_entries:
        validate_entry(entry, dataset=dataset, backbone=backbone, errors=errors)
        cache_path = entry.get("feature_cache_path")
        if isinstance(cache_path, str) and cache_path:
            try:
                inspection = inspect_feature_cache_metadata(Path(cache_path))
                cache_inspections[entry_key(entry)] = inspection
                validate_cache_inspection(entry, inspection, errors=errors)
            except Exception as exc:
                errors.append(f"failed to inspect feature cache metadata for {cache_path}: {exc}")
        else:
            errors.append(f"missing feature_cache_path for summary {entry.get('summary_path')}")

    feature_dim = infer_feature_dim(required_entries, cache_inspections)
    if feature_dim is not None and feature_dim != 512:
        errors.append(f"feature_dim must be 512 for RemoteCLIP ViT-B/32 caches, found {feature_dim}")

    num_classes = infer_num_classes(cache_inspections)
    support_counts_by_shot: dict[str, int] = {}
    if num_classes is None:
        errors.append("could not infer num_classes from feature cache metadata")
    else:
        for split_id in shot_splits:
            support_entries = [entry for entry in entries_by_split.get(split_id, []) if entry.get("split_section") == "support"]
            support_count = sum_int(entry.get("image_count") for entry in support_entries)
            support_counts_by_shot[split_id] = support_count
            shot = shot_from_split_id(split_id)
            if shot is None:
                errors.append(f"could not parse shot number from split id: {split_id}")
            elif support_count != shot * num_classes:
                errors.append(
                    f"{split_id} support image_count={support_count} does not equal shot*num_classes={shot * num_classes}"
                )

    report = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": dataset,
        "backbone": backbone,
        "base_split": base_split,
        "shot_splits": shot_splits,
        "required_base_sections_found": required_base_sections_found,
        "required_support_splits_found": required_support_splits_found,
        "num_entries_checked": len(required_entries),
        "total_images_checked": sum_int(entry.get("image_count") for entry in required_entries),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "support_counts_by_shot": support_counts_by_shot,
        "manifest_path": str(manifest_source),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "saves_predictions": False,
        "saves_logits": False,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "source_script": "scripts/check_feature_cache_consumer_preflight.py",
        "created_at": utc_now_iso(),
    }
    destination = Path(output_dir) / "feature_cache_consumer_preflight_report.json"
    report_path = safe_write_json(destination, report, overwrite=False)
    return report_path, bool(report["is_valid"])


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("entries", manifest)
    if not isinstance(entries, list):
        raise ValueError("manifest must contain an entries list")
    return [entry for entry in entries if isinstance(entry, dict)]


def resolve_entry_from_summary(entry: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    summary_value = entry.get("summary_path")
    if not isinstance(summary_value, str) or not summary_value:
        raise ValueError("manifest entry is missing summary_path")
    summary_path = Path(summary_value)
    if not summary_path.is_absolute() and not summary_path.exists():
        summary_path = manifest_dir / summary_path
    summary = read_json(summary_path)
    resolved = dict(entry)
    resolved.update(summary)
    resolved["summary_path"] = str(summary_path)
    if not resolved.get("run_dir"):
        resolved["run_dir"] = str(summary_path.parent)
    return resolved


def entry_matches_split(entry: dict[str, Any], split_id: str) -> bool:
    requested_tokens = split_tokens(split_id)
    for key in ("split_id", "split_name", "base_split", "split", "split_path", "summary_path", "run_dir", "feature_cache_path"):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        value_tokens = {value, path.name, path.stem, *path.parts}
        value_tokens.update(split_tokens(path.stem))
        if requested_tokens & value_tokens:
            return True
    return False


def split_tokens(split_id: str) -> set[str]:
    tokens = {split_id}
    if split_id.startswith("base_seed"):
        tokens.add(split_id.replace("base_seed", "base_split_seed", 1))
    if split_id.startswith("base_split_seed"):
        tokens.add(split_id.replace("base_split_seed", "base_seed", 1))
    return {token for token in tokens if token}


def validate_entry(entry: dict[str, Any], *, dataset: str, backbone: str, errors: list[str]) -> None:
    context = str(entry.get("summary_path"))
    if entry.get("dataset") != dataset:
        errors.append(f"{context}: dataset mismatch, expected {dataset}, found {entry.get('dataset')}")
    if entry.get("backbone") != backbone:
        errors.append(f"{context}: backbone mismatch, expected {backbone}, found {entry.get('backbone')}")
    if entry.get("checkpoint_loaded") is not True:
        errors.append(f"{context}: checkpoint_loaded must be true")
    if entry.get("final_weights_loaded_from_checkpoint") is not None and entry.get("final_weights_loaded_from_checkpoint") is not True:
        errors.append(f"{context}: final_weights_loaded_from_checkpoint must be true when present")
    if int_or_none(entry.get("missing_keys_count")) != 0:
        errors.append(f"{context}: missing_keys_count must be 0")
    if int_or_none(entry.get("unexpected_keys_count")) != 0:
        errors.append(f"{context}: unexpected_keys_count must be 0")
    shape = feature_shape(entry.get("feature_shape"))
    image_count = int_or_none(entry.get("image_count"))
    if len(shape) != 2:
        errors.append(f"{context}: feature_shape must have length 2")
    else:
        if image_count is None or shape[0] != image_count:
            errors.append(f"{context}: feature_shape[0] must equal image_count")
        if shape[1] != 512:
            errors.append(f"{context}: feature_shape[1] must be 512")
    for flag in CONSUMER_FORBIDDEN_TRUE_FLAGS:
        if bool(entry.get(flag)):
            errors.append(f"{context}: {flag} must be false for consumer preflight")


def inspect_feature_cache_metadata(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("feature cache file must contain a mapping")
    image_features = data.get("image_features")
    class_to_idx = data.get("class_to_idx")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return {
        "feature_shape": list(shape_of_2d(image_features)),
        "num_classes": len(class_to_idx) if isinstance(class_to_idx, dict) else None,
        "dataset": data.get("dataset", metadata.get("dataset", "")),
        "backbone": data.get("backbone", metadata.get("backbone", "")),
    }


def validate_cache_inspection(entry: dict[str, Any], inspection: dict[str, Any], *, errors: list[str]) -> None:
    context = str(entry.get("summary_path"))
    if inspection.get("dataset") and inspection.get("dataset") != entry.get("dataset"):
        errors.append(f"{context}: feature cache dataset metadata does not match summary")
    if inspection.get("backbone") and inspection.get("backbone") != entry.get("backbone"):
        errors.append(f"{context}: feature cache backbone metadata does not match summary")
    cache_shape = feature_shape(inspection.get("feature_shape"))
    summary_shape = feature_shape(entry.get("feature_shape"))
    if cache_shape and summary_shape and cache_shape != summary_shape:
        errors.append(f"{context}: feature cache shape {cache_shape} does not match summary feature_shape {summary_shape}")


def infer_feature_dim(entries: list[dict[str, Any]], cache_inspections: dict[str, dict[str, Any]]) -> int | None:
    for entry in entries:
        shape = feature_shape(entry.get("feature_shape"))
        if len(shape) == 2:
            return shape[1]
    for inspection in cache_inspections.values():
        shape = feature_shape(inspection.get("feature_shape"))
        if len(shape) == 2:
            return shape[1]
    return None


def infer_num_classes(cache_inspections: dict[str, dict[str, Any]]) -> int | None:
    for inspection in cache_inspections.values():
        value = inspection.get("num_classes")
        if isinstance(value, int) and value > 0:
            return value
    return None


def entry_key(entry: dict[str, Any]) -> str:
    return str(entry.get("summary_path") or entry.get("feature_cache_path") or id(entry))


def feature_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, (int, float))]
    if isinstance(value, tuple):
        return [int(item) for item in value if isinstance(item, (int, float))]
    return []


def shot_from_split_id(split_id: str) -> int | None:
    match = re.search(r"shot_(\d+)", split_id)
    return int(match.group(1)) if match else None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def sum_int(values: Any) -> int:
    total = 0
    for value in values:
        parsed = int_or_none(value)
        if parsed is not None:
            total += parsed
    return total


if __name__ == "__main__":
    main()
