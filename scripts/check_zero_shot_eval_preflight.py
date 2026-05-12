#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_cache import shape_of_1d, shape_of_2d, to_labels
from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


BASE_SECTIONS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only cached zero-shot evaluation input preflight.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--base-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_zero_shot_eval_preflight(
        manifest_path=args.manifest,
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"zero_shot_eval_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_zero_shot_eval_preflight(
    *,
    manifest_path: str | Path,
    dataset: str,
    backbone: str,
    base_split: str,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> tuple[Path, bool]:
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    manifest_source = Path(manifest_path)
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    manifest = read_json(manifest_source)
    entries = resolve_manifest_entries(manifest, manifest_source, errors)
    selected_entries = [
        entry for entry in entries if entry.get("dataset") == dataset and entry.get("backbone") == backbone
    ]
    base_request = make_base_request(base_split, dataset)
    base_entries = select_base_entries(selected_entries, base_request, warnings)

    image_cache_summary: dict[str, Any] = {}
    inspections: dict[str, dict[str, Any]] = {}
    reference_class_to_idx: dict[str, int] | None = None
    for section in BASE_SECTIONS:
        entry = base_entries.get(section)
        if entry is None:
            errors.append(f"{base_split} is missing required {section} feature cache")
            image_cache_summary[section] = {"cache_exists": False, "is_ready": False}
            continue
        inspection = inspect_cache_entry(entry, section, dataset, backbone, base_request, reference_class_to_idx, errors)
        if inspection.get("class_to_idx"):
            reference_class_to_idx = dict(inspection["class_to_idx"])
        inspections[section] = inspection
        image_cache_summary[section] = public_image_summary(inspection)

    feature_dim = infer_single_value([inspection.get("feature_dim") for inspection in inspections.values()], "feature_dim", errors)
    num_classes = infer_num_classes(reference_class_to_idx, inspections, errors)
    text_feature_summary = build_text_feature_summary(
        inspections=inspections,
        entries=base_entries,
        num_classes=num_classes,
        feature_dim=feature_dim,
        warnings=warnings,
        errors=errors,
    )

    val_ready = bool(image_cache_summary.get("val", {}).get("is_ready"))
    test_ready = bool(image_cache_summary.get("test", {}).get("is_ready"))
    valid_text_sections = set(text_feature_summary.get("valid_text_feature_sections", []))
    missing_eval_text_sections = [section for section in ("val", "test") if section not in valid_text_sections]
    for section in missing_eval_text_sections:
        errors.append(f"{section}: zero-shot cached evaluation requires valid text_features in the {section} cache")
    zero_shot_input_ready = not errors and not missing_eval_text_sections and val_ready and test_ready
    if missing_eval_text_sections:
        recommendations.append(
            "Generate or attach text_features to the val/test caches with shape [num_classes, feature_dim] "
            "before running cached zero-shot evaluation."
        )
    if not val_ready or not test_ready:
        recommendations.append("Fix val/test image cache readiness before running cached zero-shot evaluation.")

    report = {
        "is_valid": zero_shot_input_ready,
        "zero_shot_input_ready": zero_shot_input_ready,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "recommendations": recommendations,
        "dataset": dataset,
        "backbone": backbone,
        "seed": infer_seed(base_request),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "manifest_path": str(manifest_source),
        "checked_base_split": describe_base_request(base_request),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "text_feature_summary": text_feature_summary,
        "image_cache_summary": image_cache_summary,
        "val_ready_for_eval_input": val_ready,
        "test_ready_for_eval_input": test_ready,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "loads_model": False,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/check_zero_shot_eval_preflight.py",
    }
    output_path = unique_dir(output_root / f"{dataset}_{backbone}_{report['seed']}")
    report_path = safe_write_json(output_path / "zero_shot_eval_preflight_report.json", report)
    return report_path, bool(report["is_valid"])


def resolve_manifest_entries(manifest: dict[str, Any], manifest_path: Path, errors: list[str]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        errors.append("manifest must contain an entries list")
        return []
    resolved = []
    manifest_dir = manifest_path.parent
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            errors.append("manifest entries must be mappings")
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
        resolved.append(entry)
    return resolved


def make_base_request(value: str, dataset: str) -> dict[str, Any]:
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
    if split_id.startswith("base_seed"):
        tokens.add(split_id.replace("base_seed", "base_split_seed", 1))
    if split_id.startswith("base_split_seed"):
        tokens.add(split_id.replace("base_split_seed", "base_seed", 1))
    split_data = read_json(split_path) if split_path is not None else None
    return {
        "input": value,
        "split_id": split_id,
        "split_path": split_path,
        "split_data": split_data,
        "tokens": {token for token in tokens if token},
    }


def describe_base_request(request: dict[str, Any]) -> dict[str, Any]:
    split_data = request.get("split_data") if isinstance(request.get("split_data"), dict) else {}
    return {
        "input": request["input"],
        "split_id": request["split_id"],
        "split_path": str(request["split_path"]) if request.get("split_path") is not None else None,
        "dataset": split_data.get("dataset"),
        "seed": split_data.get("seed"),
        "num_classes": split_data.get("num_classes"),
    }


def select_base_entries(
    entries: list[dict[str, Any]], request: dict[str, Any], warnings: list[str]
) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {}
    for section in BASE_SECTIONS:
        matches = [entry for entry in entries if entry.get("split_section") == section and entry_matches_request(entry, request)]
        selected[section] = choose_entry(matches, f"{request['input']}:{section}", warnings)
    return selected


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
        if tokens & {value, path.name, path.stem, *path.parts}:
            return True
    return False


def inspect_cache_entry(
    entry: dict[str, Any],
    section: str,
    dataset: str,
    backbone: str,
    base_request: dict[str, Any],
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> dict[str, Any]:
    summary_path = str(entry.get("summary_path", ""))
    cache_path = resolve_cache_path(entry, Path(summary_path).parent if summary_path else Path("."))
    if cache_path is None:
        errors.append(f"{summary_path}: missing feature_cache_path")
        return {"cache_exists": False, "is_ready": False, "local_errors": ["missing feature_cache_path"]}
    if not cache_path.exists():
        errors.append(f"{summary_path}: feature cache file does not exist: {cache_path}")
        return {"cache_path": str(cache_path), "cache_exists": False, "is_ready": False, "local_errors": ["missing cache file"]}
    inspection = inspect_feature_cache(cache_path)
    inspection.update({"cache_path": str(cache_path), "cache_exists": True, "summary_path": summary_path, "section": section})
    validate_image_cache(
        entry=entry,
        inspection=inspection,
        section=section,
        dataset=dataset,
        backbone=backbone,
        base_request=base_request,
        reference_class_to_idx=reference_class_to_idx,
        errors=errors,
    )
    inspection["is_ready"] = not inspection["local_errors"]
    return inspection


def inspect_feature_cache(cache_path: Path) -> dict[str, Any]:
    local_errors: list[str] = []
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        return {"local_errors": ["feature cache must contain a mapping"]}
    image_features = pick_field(data, ["image_features", "features"])
    labels = pick_field(data, ["image_labels", "labels"])
    paths = pick_field(data, ["image_paths", "paths"])
    text_features = data.get("text_features")
    text_prompts = data.get("text_prompts")
    class_to_idx = data.get("class_to_idx")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    feature_shape = list(shape_of_2d(image_features))
    label_shape = list(shape_of_1d(labels))
    path_count = len(paths) if isinstance(paths, (list, tuple)) else None
    text_feature_shape = list(shape_of_2d(text_features)) if text_features is not None else []
    class_mapping = dict(class_to_idx) if isinstance(class_to_idx, dict) else {}
    label_values = labels_to_list(labels, local_errors)
    feature_dim = int_or_none(data.get("feature_dim")) or (feature_shape[1] if len(feature_shape) == 2 else None)
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
    if len(feature_shape) == 2 and len(label_shape) == 1 and feature_shape[0] != label_shape[0]:
        local_errors.append("labels length must match image_features first dimension")
    if paths is not None and not isinstance(paths, (list, tuple)):
        local_errors.append("paths must be a list/tuple")
    if len(feature_shape) == 2 and path_count is not None and path_count != feature_shape[0]:
        local_errors.append("paths length must match image_features first dimension")
    if len(feature_shape) == 2 and feature_dim is not None and feature_shape[1] != feature_dim:
        local_errors.append(f"feature_dim={feature_dim} does not match image feature width={feature_shape[1]}")
    if class_mapping:
        values = sorted(int(value) for value in class_mapping.values())
        if values != list(range(len(values))):
            local_errors.append("class_to_idx values must be contiguous from 0 to num_classes-1")
        if any(label < 0 or label >= len(class_mapping) for label in label_values):
            local_errors.append("labels contain values outside class_to_idx range")
    return {
        "local_errors": local_errors,
        "dataset": data.get("dataset", metadata.get("dataset")),
        "backbone": data.get("backbone", metadata.get("backbone")),
        "feature_shape": feature_shape,
        "label_shape": label_shape,
        "path_count": path_count,
        "feature_dim": feature_dim,
        "num_classes": len(class_mapping) if class_mapping else None,
        "class_to_idx": class_mapping,
        "label_min": min(label_values) if label_values else None,
        "label_max": max(label_values) if label_values else None,
        "has_image_features": image_features is not None,
        "has_labels": labels is not None,
        "has_paths": paths is not None,
        "has_class_to_idx": bool(class_mapping),
        "has_text_features": text_features is not None,
        "text_feature_shape": text_feature_shape,
        "has_text_prompts": isinstance(text_prompts, list) and bool(text_prompts),
        "num_text_prompts": len(text_prompts) if isinstance(text_prompts, list) else 0,
        "text_class_names": text_class_names(data, metadata),
        "prompt_metadata_keys": sorted(key for key in metadata if "prompt" in str(key).lower()),
    }


def validate_image_cache(
    *,
    entry: dict[str, Any],
    inspection: dict[str, Any],
    section: str,
    dataset: str,
    backbone: str,
    base_request: dict[str, Any],
    reference_class_to_idx: dict[str, int] | None,
    errors: list[str],
) -> None:
    context = str(entry.get("summary_path") or inspection.get("cache_path"))
    for local_error in inspection["local_errors"]:
        errors.append(f"{context}: {local_error}")
    if entry.get("dataset") != dataset:
        errors.append(f"{context}: summary dataset mismatch, expected {dataset}, found {entry.get('dataset')}")
    if entry.get("backbone") != backbone:
        errors.append(f"{context}: summary backbone mismatch, expected {backbone}, found {entry.get('backbone')}")
    if entry.get("split_section") != section:
        errors.append(f"{context}: split_section mismatch, expected {section}, found {entry.get('split_section')}")
    if inspection.get("dataset") and inspection.get("dataset") != dataset:
        errors.append(f"{context}: cache dataset metadata mismatch")
    if inspection.get("backbone") and inspection.get("backbone") != backbone:
        errors.append(f"{context}: cache backbone metadata mismatch")
    summary_shape = shape_list(entry.get("feature_shape"))
    if summary_shape and inspection.get("feature_shape") and summary_shape != inspection["feature_shape"]:
        errors.append(f"{context}: cache feature_shape {inspection['feature_shape']} does not match summary {summary_shape}")
    if reference_class_to_idx is not None and inspection.get("class_to_idx") != reference_class_to_idx:
        errors.append(f"{context}: class_to_idx does not match previously checked cache")
    split_data = base_request.get("split_data")
    if isinstance(split_data, dict):
        split_class_to_idx = split_data.get("class_to_idx")
        if isinstance(split_class_to_idx, dict) and inspection.get("class_to_idx") != split_class_to_idx:
            errors.append(f"{context}: cache class_to_idx does not match base split file")
        split_section = split_data.get(section)
        if isinstance(split_section, list) and inspection.get("feature_shape"):
            if len(split_section) != inspection["feature_shape"][0]:
                errors.append(
                    f"{context}: cache sample count {inspection['feature_shape'][0]} does not match split {section} length {len(split_section)}"
                )


def build_text_feature_summary(
    *,
    inspections: dict[str, dict[str, Any]],
    entries: dict[str, dict[str, Any] | None],
    num_classes: int | None,
    feature_dim: int | None,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    by_section: dict[str, Any] = {}
    valid_sections = []
    manifest_text_shapes = {}
    for section, entry in entries.items():
        if isinstance(entry, dict):
            shape = shape_list(entry.get("text_feature_shape") or entry.get("text_features_shape"))
            if shape:
                manifest_text_shapes[section] = shape
    for section, inspection in inspections.items():
        text_errors = []
        has_text = bool(inspection.get("has_text_features"))
        shape = inspection.get("text_feature_shape", [])
        if not has_text:
            text_errors.append("text_features missing from cache")
        elif num_classes is not None and feature_dim is not None and shape != [num_classes, feature_dim]:
            text_errors.append(f"text_features shape {shape} does not equal [num_classes, feature_dim]=[{num_classes}, {feature_dim}]")
        class_order = class_order_summary(inspection, num_classes, warnings)
        prompts = prompt_summary(inspection, num_classes, warnings)
        section_summary = {
            "has_text_features": has_text,
            "text_feature_shape": shape,
            "manifest_text_feature_shape": manifest_text_shapes.get(section),
            "shape_valid": bool(has_text and not text_errors),
            "class_order": class_order,
            "prompt_metadata": prompts,
            "errors": text_errors,
        }
        if text_errors:
            for error in text_errors:
                if has_text:
                    errors.append(f"{section}: {error}")
        else:
            valid_sections.append(section)
        by_section[section] = section_summary
    if not valid_sections:
        errors.append("zero-shot cached evaluation requires text_features in at least one base cache")
    return {
        "has_valid_text_features": bool(valid_sections),
        "valid_text_feature_sections": valid_sections,
        "by_section": by_section,
        "manifest_text_feature_shapes": manifest_text_shapes,
    }


def class_order_summary(inspection: dict[str, Any], num_classes: int | None, warnings: list[str]) -> dict[str, Any]:
    class_to_idx = inspection.get("class_to_idx") if isinstance(inspection.get("class_to_idx"), dict) else {}
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))]
    text_class_names = inspection.get("text_class_names")
    if isinstance(text_class_names, list) and text_class_names:
        aligned = text_class_names[: len(class_names)] == class_names
        return {
            "class_to_idx_order": class_names,
            "text_class_names": text_class_names,
            "alignment_checked": True,
            "aligned": aligned,
        }
    if num_classes is not None and len(class_names) == num_classes:
        warnings.append("text class names are not stored; assuming text_features follow class_to_idx index order")
    return {
        "class_to_idx_order": class_names,
        "text_class_names": [],
        "alignment_checked": False,
        "aligned": len(class_names) == num_classes if num_classes is not None else False,
        "alignment_assumption": "text_features rows must follow class_to_idx index order",
    }


def prompt_summary(inspection: dict[str, Any], num_classes: int | None, warnings: list[str]) -> dict[str, Any]:
    num_prompts = int_or_none(inspection.get("num_text_prompts")) or 0
    has_prompts = bool(inspection.get("has_text_prompts"))
    valid = False
    if num_classes is not None and has_prompts:
        valid = num_prompts == num_classes or (num_prompts > num_classes and num_prompts % num_classes == 0)
        if not valid:
            warnings.append(f"text_prompts count {num_prompts} does not match num_classes or a class-template multiple")
    if not has_prompts:
        warnings.append("text prompt metadata is not stored in cache")
    return {
        "has_text_prompts": has_prompts,
        "num_text_prompts": num_prompts,
        "prompt_metadata_keys": inspection.get("prompt_metadata_keys", []),
        "prompt_count_valid": valid,
    }


def public_image_summary(inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_path": inspection.get("cache_path"),
        "cache_exists": inspection.get("cache_exists", False),
        "is_ready": inspection.get("is_ready", False),
        "feature_shape": inspection.get("feature_shape"),
        "label_shape": inspection.get("label_shape"),
        "path_count": inspection.get("path_count"),
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


def text_class_names(data: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    for key in ("text_class_names", "class_names_for_text", "text_label_names"):
        value = data.get(key, metadata.get(key))
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def infer_single_value(values: list[Any], name: str, errors: list[str]) -> int | None:
    parsed = sorted({int(value) for value in values if isinstance(value, int)})
    if not parsed:
        errors.append(f"could not infer {name} from checked caches")
        return None
    if len(parsed) > 1:
        errors.append(f"{name} is inconsistent across checked caches: {parsed}")
        return None
    return parsed[0]


def infer_num_classes(
    reference_class_to_idx: dict[str, int] | None,
    inspections: dict[str, dict[str, Any]],
    errors: list[str],
) -> int | None:
    if reference_class_to_idx:
        return len(reference_class_to_idx)
    return infer_single_value([inspection.get("num_classes") for inspection in inspections.values()], "num_classes", errors)


def infer_seed(request: dict[str, Any]) -> str:
    split_data = request.get("split_data")
    if isinstance(split_data, dict) and split_data.get("seed") is not None:
        return f"seed{split_data['seed']}"
    match = re.search(r"seed(\d+)", request["split_id"])
    if match:
        return f"seed{match.group(1)}"
    return "seed_unknown"


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


def shape_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if isinstance(item, (int, float))]
    return []


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
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


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


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("zero-shot evaluation preflight reports must not be written under results/raw")


if __name__ == "__main__":
    main()
