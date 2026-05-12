#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_cache import shape_of_1d, shape_of_2d, to_labels
from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


BASE_SECTIONS = ["train", "val", "test"]
SUPPORTED_METHODS = ["tip_adapter", "proto_adapter", "rs_cpc"]
RS_CPC_M_VALUES = [1, 2, 4, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only adapter/RS-CPC feature-cache input preflight.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--base-split", required=True)
    parser.add_argument("--shot-splits", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_adapter_input_preflight(
        manifest_path=args.manifest,
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split,
        shot_splits=args.shot_splits,
        methods=args.methods,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"adapter_input_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_adapter_input_preflight(
    *,
    manifest_path: str | Path,
    dataset: str,
    backbone: str,
    base_split: str,
    shot_splits: list[str],
    methods: list[str],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> tuple[Path, bool]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_source = Path(manifest_path)

    unknown_methods = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown_methods:
        errors.append(f"unsupported methods: {unknown_methods}")
    checked_methods = [method for method in methods if method in SUPPORTED_METHODS]

    manifest = read_json(manifest_source)
    entries = resolve_manifest_entries(manifest, manifest_source, errors)
    selected_entries = [
        entry for entry in entries if entry.get("dataset") == dataset and entry.get("backbone") == backbone
    ]

    base_request = make_split_request(base_split, dataset)
    shot_requests = [make_split_request(split, dataset) for split in shot_splits]
    checked_base_split = describe_split_request(base_request)
    checked_shot_splits = [describe_split_request(request) for request in shot_requests]

    base_sections = select_base_sections(selected_entries, base_request, warnings)
    shot_support = {
        request["input"]: select_shot_support(selected_entries, request, warnings) for request in shot_requests
    }

    per_split_summary: dict[str, Any] = {}
    all_inspections: list[dict[str, Any]] = []
    reference_class_to_idx: dict[str, int] | None = None

    base_summary, base_inspections, reference_class_to_idx = inspect_base_split(
        base_request=base_request,
        base_sections=base_sections,
        dataset=dataset,
        backbone=backbone,
        reference_class_to_idx=reference_class_to_idx,
        errors=errors,
    )
    per_split_summary[base_request["input"]] = base_summary
    all_inspections.extend(base_inspections)

    shot_summaries: dict[str, Any] = {}
    for request in shot_requests:
        summary, inspections, reference_class_to_idx = inspect_shot_split(
            shot_request=request,
            support_entry=shot_support[request["input"]],
            dataset=dataset,
            backbone=backbone,
            reference_class_to_idx=reference_class_to_idx,
            errors=errors,
        )
        per_split_summary[request["input"]] = summary
        shot_summaries[request["input"]] = summary
        all_inspections.extend(inspections)

    feature_dim = infer_single_feature_dim(all_inspections, errors)
    if backbone == "remoteclip_vit_b32" and feature_dim is not None and feature_dim != 512:
        errors.append(f"feature_dim must be 512 for remoteclip_vit_b32, found {feature_dim}")
    num_classes = infer_num_classes(reference_class_to_idx, all_inspections, errors)

    per_method_input_summary = build_method_input_summary(
        methods=checked_methods,
        shot_summaries=shot_summaries,
        num_classes=num_classes,
        warnings=warnings,
    )

    report = {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": dataset,
        "backbone": backbone,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "manifest_path": str(manifest_source),
        "checked_base_split": checked_base_split,
        "checked_shot_splits": checked_shot_splits,
        "checked_methods": checked_methods,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "per_split_summary": per_split_summary,
        "per_method_input_summary": per_method_input_summary,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "saves_predictions": False,
        "saves_logits": False,
        "source_script": "scripts/check_adapter_input_preflight.py",
        "rs_cpc_m_gt_shot_policy": (
            "Each default M is marked ready only when M <= the minimum per-class support count. "
            "If any default M is too large, that M gets method_input_ready_by_M=false and a warning; "
            "the shot-level RS-CPC method_input_ready is false until all default M values are feasible."
        ),
    }

    destination_dir = report_output_dir(Path(output_dir), dataset, backbone, base_request)
    report_path = safe_write_json(unique_path(destination_dir / "adapter_input_preflight_report.json"), report)
    return report_path, bool(report["is_valid"])


def resolve_manifest_entries(manifest: dict[str, Any], manifest_path: Path, errors: list[str]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        errors.append("manifest must contain an entries list")
        return []

    resolved_entries = []
    manifest_dir = manifest_path.parent
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            warnings_context = f"manifest entry at index {len(resolved_entries)} is not a mapping"
            errors.append(warnings_context)
            continue
        entry = dict(raw_entry)
        summary_path = resolve_path(entry.get("summary_path"), manifest_dir)
        if summary_path is not None and summary_path.exists():
            try:
                summary = read_json(summary_path)
                merged = dict(entry)
                merged.update(summary)
                entry = merged
            except Exception as exc:
                errors.append(f"failed to read feature extraction summary {summary_path}: {exc}")
        elif summary_path is not None:
            errors.append(f"feature extraction summary does not exist: {summary_path}")
        entry["summary_path"] = str(summary_path) if summary_path is not None else str(entry.get("summary_path", ""))
        cache_path = resolve_cache_path(entry, manifest_dir)
        if cache_path is not None:
            entry["feature_cache_path"] = str(cache_path)
        resolved_entries.append(entry)
    return resolved_entries


def make_split_request(value: str, dataset: str) -> dict[str, Any]:
    path = Path(value)
    candidates = [path]
    if not path.exists() and not path.suffix:
        candidates.append(Path("splits") / dataset / f"{value}.json")
        if value.startswith("base_seed"):
            candidates.append(Path("splits") / dataset / f"{value.replace('base_seed', 'base_split_seed', 1)}.json")
        if value.startswith("base_split_seed"):
            candidates.append(Path("splits") / dataset / f"{value.replace('base_split_seed', 'base_seed', 1)}.json")
    split_path = next((candidate for candidate in candidates if candidate.exists()), None)
    split_id = path.stem if path.suffix else value
    tokens = {value, path.name, path.stem, split_id}
    if split_id.startswith("base_split_seed"):
        tokens.add(split_id.replace("base_split_seed", "base_seed", 1))
    if split_id.startswith("base_seed"):
        tokens.add(split_id.replace("base_seed", "base_split_seed", 1))
    tokens = {token for token in tokens if token}
    split_data = read_json(split_path) if split_path is not None else None
    return {
        "input": value,
        "split_id": split_id,
        "split_path": split_path,
        "split_data": split_data,
        "tokens": tokens,
    }


def describe_split_request(request: dict[str, Any]) -> dict[str, Any]:
    split_data = request.get("split_data") if isinstance(request.get("split_data"), dict) else {}
    return {
        "input": request["input"],
        "split_id": request["split_id"],
        "split_path": str(request["split_path"]) if request.get("split_path") is not None else None,
        "dataset": split_data.get("dataset"),
        "seed": split_data.get("seed"),
        "shot": split_data.get("shot"),
        "num_classes": split_data.get("num_classes"),
    }


def select_base_sections(
    entries: list[dict[str, Any]], request: dict[str, Any], warnings: list[str]
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for section in BASE_SECTIONS:
        matches = [entry for entry in entries if entry.get("split_section") == section and entry_matches_request(entry, request)]
        result[section] = choose_entry(matches, f"{request['input']}:{section}", warnings)
    return result


def select_shot_support(
    entries: list[dict[str, Any]], request: dict[str, Any], warnings: list[str]
) -> dict[str, Any] | None:
    matches = [entry for entry in entries if entry.get("split_section") == "support" and entry_matches_request(entry, request)]
    return choose_entry(matches, f"{request['input']}:support", warnings)


def choose_entry(matches: list[dict[str, Any]], context: str, warnings: list[str]) -> dict[str, Any] | None:
    if not matches:
        return None
    matches = sorted(matches, key=lambda entry: str(entry.get("summary_path", "")))
    if len(matches) > 1:
        warnings.append(f"{context} has {len(matches)} matching manifest entries; using the first sorted summary path")
    return matches[0]


def entry_matches_request(entry: dict[str, Any], request: dict[str, Any]) -> bool:
    tokens = request["tokens"]
    for key in ("summary_path", "run_dir", "feature_cache_path", "split_path"):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        path_tokens = {value, path.name, path.stem, *path.parts}
        if tokens & path_tokens:
            return True
    return False


def inspect_base_split(
    *,
    base_request: dict[str, Any],
    base_sections: dict[str, dict[str, Any] | None],
    dataset: str,
    backbone: str,
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int] | None]:
    sections: dict[str, Any] = {}
    inspections: list[dict[str, Any]] = []
    for section in BASE_SECTIONS:
        entry = base_sections.get(section)
        if entry is None:
            errors.append(f"{base_request['input']} is missing required {section} feature cache")
            sections[section] = {"cache_exists": False, "is_ready": False}
            continue
        inspection = inspect_and_validate_entry(
            entry=entry,
            expected_section=section,
            dataset=dataset,
            backbone=backbone,
            split_request=base_request,
            reference_class_to_idx=reference_class_to_idx,
            errors=errors,
        )
        if inspection.get("class_to_idx"):
            reference_class_to_idx = dict(inspection["class_to_idx"])
        inspections.append(inspection)
        sections[section] = public_inspection_summary(inspection)
    return (
        {
            "split_kind": "base",
            "split_id": base_request["split_id"],
            "split_path": str(base_request["split_path"]) if base_request.get("split_path") is not None else None,
            "sections": sections,
            "has_train_cache": bool(sections.get("train", {}).get("is_ready")),
            "val_ready_for_tuning_input": bool(sections.get("val", {}).get("is_ready")),
            "test_ready_for_evaluation_input": bool(sections.get("test", {}).get("is_ready")),
            "performs_tuning": False,
            "performs_evaluation": False,
        },
        inspections,
        reference_class_to_idx,
    )


def inspect_shot_split(
    *,
    shot_request: dict[str, Any],
    support_entry: dict[str, Any] | None,
    dataset: str,
    backbone: str,
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int] | None]:
    shot = shot_from_request(shot_request)
    if support_entry is None:
        errors.append(f"{shot_request['input']} is missing required support feature cache")
        return (
            {
                "split_kind": "shot",
                "split_id": shot_request["split_id"],
                "split_path": str(shot_request["split_path"]) if shot_request.get("split_path") is not None else None,
                "shot": shot,
                "support": {"cache_exists": False, "is_ready": False},
                "support_balanced": False,
                "min_support_per_class": 0,
            },
            [],
            reference_class_to_idx,
        )

    inspection = inspect_and_validate_entry(
        entry=support_entry,
        expected_section="support",
        dataset=dataset,
        backbone=backbone,
        split_request=shot_request,
        reference_class_to_idx=reference_class_to_idx,
        errors=errors,
    )
    if inspection.get("class_to_idx"):
        reference_class_to_idx = dict(inspection["class_to_idx"])
    support_counts = inspection.get("label_counts", {})
    num_classes = inspection.get("num_classes")
    expected_total = shot * num_classes if isinstance(shot, int) and isinstance(num_classes, int) else None
    support_balanced = False
    min_support_per_class = 0
    if isinstance(shot, int) and isinstance(num_classes, int):
        expected_labels = [str(label) for label in range(num_classes)]
        missing_labels = [label for label in expected_labels if int(support_counts.get(label, 0)) == 0]
        bad_counts = {
            label: count
            for label, count in support_counts.items()
            if 0 <= int(label) < num_classes and int(count) != shot
        }
        if missing_labels:
            errors.append(f"{shot_request['input']} support cache is missing labels: {missing_labels}")
        if bad_counts:
            errors.append(f"{shot_request['input']} support per-class counts do not match shot={shot}: {bad_counts}")
        if expected_total is not None and inspection.get("num_samples") != expected_total:
            errors.append(
                f"{shot_request['input']} support sample count={inspection.get('num_samples')} "
                f"does not equal shot*num_classes={expected_total}"
            )
        counts = [int(support_counts.get(str(label), 0)) for label in range(num_classes)]
        support_balanced = all(count == shot for count in counts)
        min_support_per_class = min(counts) if counts else 0
    else:
        errors.append(f"could not infer shot or num_classes for {shot_request['input']}")

    validate_split_file_against_support(shot_request, inspection, shot, errors)

    summary = {
        "split_kind": "shot",
        "split_id": shot_request["split_id"],
        "split_path": str(shot_request["split_path"]) if shot_request.get("split_path") is not None else None,
        "shot": shot,
        "support": public_inspection_summary(inspection),
        "support_counts_by_label": support_counts,
        "support_balanced": support_balanced,
        "min_support_per_class": min_support_per_class,
        "expected_support_entries": expected_total,
    }
    return summary, [inspection], reference_class_to_idx


def inspect_and_validate_entry(
    *,
    entry: dict[str, Any],
    expected_section: str,
    dataset: str,
    backbone: str,
    split_request: dict[str, Any],
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> dict[str, Any]:
    summary_path = str(entry.get("summary_path", ""))
    if entry.get("dataset") != dataset:
        errors.append(f"{summary_path}: dataset mismatch, expected {dataset}, found {entry.get('dataset')}")
    if entry.get("backbone") != backbone:
        errors.append(f"{summary_path}: backbone mismatch, expected {backbone}, found {entry.get('backbone')}")
    if entry.get("split_section") != expected_section:
        errors.append(
            f"{summary_path}: split_section mismatch, expected {expected_section}, found {entry.get('split_section')}"
        )

    cache_path = resolve_cache_path(entry, Path(summary_path).parent if summary_path else Path("."))
    if cache_path is None:
        errors.append(f"{summary_path}: missing feature_cache_path")
        return {"summary_path": summary_path, "cache_path": None, "cache_exists": False, "is_ready": False}
    if not cache_path.exists():
        errors.append(f"{summary_path}: feature cache file does not exist: {cache_path}")
        return {
            "summary_path": summary_path,
            "cache_path": str(cache_path),
            "cache_exists": False,
            "is_ready": False,
        }

    inspection = inspect_feature_cache(cache_path)
    inspection.update(
        {
            "summary_path": summary_path,
            "cache_path": str(cache_path),
            "cache_exists": True,
            "split_section": expected_section,
        }
    )

    validate_cache_inspection(
        entry=entry,
        inspection=inspection,
        expected_section=expected_section,
        dataset=dataset,
        backbone=backbone,
        split_request=split_request,
        reference_class_to_idx=reference_class_to_idx,
        errors=errors,
    )
    inspection["is_ready"] = not inspection.get("local_errors")
    return inspection


def inspect_feature_cache(cache_path: Path) -> dict[str, Any]:
    local_errors: list[str] = []
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        return {"local_errors": ["feature cache file must contain a mapping"]}

    image_features = pick_field(data, ["image_features", "features"])
    labels = pick_field(data, ["image_labels", "labels"])
    paths = pick_field(data, ["image_paths", "paths"])
    class_to_idx = data.get("class_to_idx")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}

    feature_shape = list(shape_of_2d(image_features))
    label_shape = list(shape_of_1d(labels))
    path_count = len(paths) if isinstance(paths, (list, tuple)) else None
    label_values = labels_to_list(labels, local_errors)
    class_mapping = dict(class_to_idx) if isinstance(class_to_idx, dict) else {}
    label_counts = Counter(label_values)
    num_classes = len(class_mapping) if class_mapping else None
    feature_dim = int(data.get("feature_dim", feature_shape[1] if len(feature_shape) == 2 else 0) or 0)

    for field_name, value in [
        ("image_features", image_features),
        ("labels", labels),
        ("paths", paths),
        ("class_to_idx", class_to_idx),
    ]:
        if value is None or (field_name == "class_to_idx" and value == {}):
            local_errors.append(f"missing required cache field: {field_name}")

    if len(feature_shape) != 2:
        local_errors.append("image_features must be a 2D tensor/list")
    if len(label_shape) != 1:
        local_errors.append("labels must be a 1D tensor/list")
    if paths is not None and not isinstance(paths, (list, tuple)):
        local_errors.append("paths must be a list/tuple")
    if len(feature_shape) == 2 and len(label_shape) == 1 and feature_shape[0] != label_shape[0]:
        local_errors.append("labels length must match image_features first dimension")
    if len(feature_shape) == 2 and path_count is not None and path_count != feature_shape[0]:
        local_errors.append("paths length must match image_features first dimension")
    if len(feature_shape) == 2 and feature_dim != feature_shape[1]:
        local_errors.append(f"feature_dim={feature_dim} does not match image feature width={feature_shape[1]}")
    if class_mapping:
        values = sorted(int(value) for value in class_mapping.values())
        if values != list(range(len(values))):
            local_errors.append("class_to_idx values must be contiguous from 0 to num_classes-1")
        if any(label < 0 or label >= len(class_mapping) for label in label_values):
            local_errors.append("labels contain values outside class_to_idx range")

    return {
        "feature_shape": feature_shape,
        "label_shape": label_shape,
        "path_count": path_count,
        "num_samples": feature_shape[0] if len(feature_shape) == 2 else None,
        "feature_dim": feature_dim if feature_dim else None,
        "num_classes": num_classes,
        "class_to_idx": class_mapping,
        "label_min": min(label_values) if label_values else None,
        "label_max": max(label_values) if label_values else None,
        "label_counts": {str(label): int(count) for label, count in sorted(label_counts.items())},
        "dataset": data.get("dataset", metadata.get("dataset")),
        "backbone": data.get("backbone", metadata.get("backbone")),
        "split_name": data.get("split_name", metadata.get("split_section")),
        "has_image_features": image_features is not None,
        "has_labels": labels is not None,
        "has_paths": paths is not None,
        "has_class_to_idx": isinstance(class_to_idx, dict) and bool(class_to_idx),
        "local_errors": local_errors,
    }


def validate_cache_inspection(
    *,
    entry: dict[str, Any],
    inspection: dict[str, Any],
    expected_section: str,
    dataset: str,
    backbone: str,
    split_request: dict[str, Any],
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> None:
    context = str(entry.get("summary_path") or inspection.get("cache_path"))
    for local_error in inspection.get("local_errors", []):
        errors.append(f"{context}: {local_error}")

    if inspection.get("dataset") and inspection.get("dataset") != dataset:
        errors.append(f"{context}: feature cache dataset metadata does not match requested dataset")
    if inspection.get("backbone") and inspection.get("backbone") != backbone:
        errors.append(f"{context}: feature cache backbone metadata does not match requested backbone")

    summary_shape = feature_shape(entry.get("feature_shape"))
    if summary_shape and inspection.get("feature_shape") and summary_shape != inspection["feature_shape"]:
        errors.append(f"{context}: cache feature_shape {inspection['feature_shape']} does not match summary {summary_shape}")
    image_count = int_or_none(entry.get("image_count"))
    if image_count is not None and inspection.get("num_samples") is not None and image_count != inspection["num_samples"]:
        errors.append(f"{context}: cache sample count {inspection['num_samples']} does not match summary image_count {image_count}")

    if reference_class_to_idx is not None and inspection.get("class_to_idx") != reference_class_to_idx:
        errors.append(f"{context}: class_to_idx does not match previously checked caches")

    split_data = split_request.get("split_data")
    if isinstance(split_data, dict):
        split_class_to_idx = split_data.get("class_to_idx")
        if isinstance(split_class_to_idx, dict) and inspection.get("class_to_idx") != split_class_to_idx:
            errors.append(f"{context}: cache class_to_idx does not match split file class_to_idx")
        split_section = split_data.get(expected_section)
        if isinstance(split_section, list) and inspection.get("num_samples") is not None:
            if len(split_section) != inspection["num_samples"]:
                errors.append(
                    f"{context}: cache sample count {inspection['num_samples']} does not match "
                    f"split {expected_section} length {len(split_section)}"
                )


def validate_split_file_against_support(
    shot_request: dict[str, Any], inspection: dict[str, Any], shot: int | None, errors: list[str]
) -> None:
    split_data = shot_request.get("split_data")
    if not isinstance(split_data, dict):
        return
    split_shot = split_data.get("shot")
    if isinstance(shot, int) and split_shot is not None and int(split_shot) != shot:
        errors.append(f"{shot_request['input']} parsed shot={shot} does not match split file shot={split_shot}")
    support_rows = split_data.get("support")
    if not isinstance(support_rows, list):
        errors.append(f"{shot_request['input']} split file must contain a support list")
        return
    split_counts = Counter(int(row["label"]) for row in support_rows if isinstance(row, dict) and "label" in row)
    cache_counts = {int(label): int(count) for label, count in inspection.get("label_counts", {}).items()}
    if dict(sorted(split_counts.items())) != dict(sorted(cache_counts.items())):
        errors.append(f"{shot_request['input']} cache label counts do not match split support label counts")


def build_method_input_summary(
    *,
    methods: list[str],
    shot_summaries: dict[str, Any],
    num_classes: int | None,
    warnings: list[str],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for method in methods:
        if method == "tip_adapter":
            summaries[method] = build_tip_summary(shot_summaries, num_classes)
        elif method == "proto_adapter":
            summaries[method] = build_proto_summary(shot_summaries, num_classes)
        elif method == "rs_cpc":
            summaries[method] = build_rs_cpc_summary(shot_summaries, num_classes, warnings)
    return summaries


def build_tip_summary(shot_summaries: dict[str, Any], num_classes: int | None) -> dict[str, Any]:
    per_shot = {}
    for split_id, summary in shot_summaries.items():
        shot = summary.get("shot")
        support = summary.get("support", {})
        expected_entries = num_classes * shot if isinstance(num_classes, int) and isinstance(shot, int) else None
        per_shot[split_id] = {
            "method_input_ready": bool(support.get("is_ready")) and summary.get("support_balanced") is True,
            "shot": shot,
            "expected_cache_entries": expected_entries,
            "actual_support_entries": support.get("num_samples"),
        }
    return {"per_shot": per_shot}


def build_proto_summary(shot_summaries: dict[str, Any], num_classes: int | None) -> dict[str, Any]:
    per_shot = {}
    for split_id, summary in shot_summaries.items():
        support = summary.get("support", {})
        per_shot[split_id] = {
            "method_input_ready": bool(support.get("is_ready")) and summary.get("support_balanced") is True,
            "shot": summary.get("shot"),
            "expected_cache_entries": num_classes,
            "actual_support_entries": support.get("num_samples"),
        }
    return {"per_shot": per_shot}


def build_rs_cpc_summary(
    shot_summaries: dict[str, Any], num_classes: int | None, warnings: list[str]
) -> dict[str, Any]:
    per_shot = {}
    for split_id, summary in shot_summaries.items():
        support = summary.get("support", {})
        min_support = int(summary.get("min_support_per_class") or 0)
        ready_by_m = {}
        expected_by_m = {}
        for m_value in RS_CPC_M_VALUES:
            ready_by_m[str(m_value)] = bool(support.get("is_ready")) and summary.get("support_balanced") is True and m_value <= min_support
            expected_by_m[str(m_value)] = num_classes * m_value if isinstance(num_classes, int) else None
            if m_value > min_support:
                warnings.append(
                    f"{split_id}: rs_cpc M={m_value} exceeds min per-class support count {min_support}; "
                    "method_input_ready_by_M is false"
                )
        per_shot[split_id] = {
            "method_input_ready": all(ready_by_m.values()) if ready_by_m else False,
            "method_input_ready_by_M": ready_by_m,
            "shot": summary.get("shot"),
            "min_support_per_class": min_support,
            "expected_cache_entries_by_M": expected_by_m,
            "actual_support_entries": support.get("num_samples"),
        }
    return {"m_values": RS_CPC_M_VALUES, "per_shot": per_shot}


def public_inspection_summary(inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_path": inspection.get("cache_path"),
        "cache_exists": inspection.get("cache_exists", False),
        "is_ready": inspection.get("is_ready", False),
        "feature_shape": inspection.get("feature_shape"),
        "label_shape": inspection.get("label_shape"),
        "path_count": inspection.get("path_count"),
        "num_samples": inspection.get("num_samples"),
        "feature_dim": inspection.get("feature_dim"),
        "num_classes": inspection.get("num_classes"),
        "label_min": inspection.get("label_min"),
        "label_max": inspection.get("label_max"),
        "has_image_features": inspection.get("has_image_features", False),
        "has_labels": inspection.get("has_labels", False),
        "has_paths": inspection.get("has_paths", False),
        "has_class_to_idx": inspection.get("has_class_to_idx", False),
        "local_errors": inspection.get("local_errors", []),
    }


def pick_field(data: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def labels_to_list(labels: Any, local_errors: list[str]) -> list[int]:
    if labels is None:
        return []
    try:
        return to_labels(labels)
    except Exception as exc:
        local_errors.append(f"could not parse labels: {exc}")
        return []


def infer_single_feature_dim(inspections: list[dict[str, Any]], errors: list[str]) -> int | None:
    dims = sorted({int(inspection["feature_dim"]) for inspection in inspections if isinstance(inspection.get("feature_dim"), int)})
    if not dims:
        errors.append("could not infer feature_dim from checked caches")
        return None
    if len(dims) > 1:
        errors.append(f"feature_dim is inconsistent across checked caches: {dims}")
        return None
    return dims[0]


def infer_num_classes(
    reference_class_to_idx: dict[str, int] | None, inspections: list[dict[str, Any]], errors: list[str]
) -> int | None:
    if reference_class_to_idx:
        return len(reference_class_to_idx)
    class_counts = sorted(
        {int(inspection["num_classes"]) for inspection in inspections if isinstance(inspection.get("num_classes"), int)}
    )
    if not class_counts:
        errors.append("could not infer num_classes from checked caches")
        return None
    if len(class_counts) > 1:
        errors.append(f"num_classes is inconsistent across checked caches: {class_counts}")
        return None
    return class_counts[0]


def shot_from_request(request: dict[str, Any]) -> int | None:
    split_data = request.get("split_data")
    if isinstance(split_data, dict) and isinstance(split_data.get("shot"), int):
        return int(split_data["shot"])
    match = re.search(r"shot_(\d+)", request["split_id"])
    return int(match.group(1)) if match else None


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


def resolve_cache_path(entry: dict[str, Any], base_dir: Path) -> Path | None:
    path = resolve_path(entry.get("feature_cache_path"), base_dir)
    if path is not None:
        return path
    run_dir = resolve_path(entry.get("run_dir"), base_dir)
    if run_dir is not None:
        return run_dir / "feature_cache.pt"
    return None


def resolve_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    return path


def report_output_dir(output_dir: Path, dataset: str, backbone: str, base_request: dict[str, Any]) -> Path:
    seed = infer_seed_label(base_request)
    expected_name = f"{dataset}_{backbone}_{seed}"
    if output_dir.name == expected_name:
        return output_dir
    return output_dir / expected_name


def infer_seed_label(request: dict[str, Any]) -> str:
    split_data = request.get("split_data")
    if isinstance(split_data, dict) and split_data.get("seed") is not None:
        return f"seed{split_data['seed']}"
    match = re.search(r"seed(\d+)", request["split_id"])
    if match:
        return f"seed{match.group(1)}"
    return "seed_unknown"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        indexed = path.with_name(f"{path.stem}_{stamp}_{index}{path.suffix}")
        if not indexed.exists():
            return indexed
    raise FileExistsError(f"Could not find non-existing output path for {path}")


if __name__ == "__main__":
    main()
