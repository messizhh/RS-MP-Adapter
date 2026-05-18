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

from src.config.config_loader import ConfigError, load_yaml_config
from src.features.feature_cache import shape_of_2d
from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


BASE_SECTIONS = ["train", "val", "test"]
TEXT_SECTION_NAMES = {"text", "text_features", "text_feature_cache"}
DEFAULT_PROMPT_TEMPLATES = ["a satellite photo of {}."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only text feature cache readiness preflight.")
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
    report_path, is_valid = run_text_feature_cache_preflight(
        manifest_path=args.manifest,
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    print(f"text_feature_cache_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_text_feature_cache_preflight(
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
    split_data = base_request.get("split_data") if isinstance(base_request.get("split_data"), dict) else None
    if split_data is None:
        errors.append(f"base split file could not be resolved or read: {base_split}")
        split_data = {}
    elif split_data.get("dataset") not in (None, dataset):
        errors.append(f"base split dataset mismatch, expected {dataset}, found {split_data.get('dataset')}")

    class_to_idx, class_names = inspect_class_mapping(split_data, errors)
    prompt_templates, prompt_template_source = load_prompt_templates(warnings, errors)
    expected_feature_dim = infer_expected_feature_dim(
        entries=selected_entries,
        base_request=base_request,
        backbone=backbone,
        warnings=warnings,
    )
    if expected_feature_dim is None:
        errors.append("could not infer expected_feature_dim from manifest entries or backbone config")

    base_feature_dir = infer_base_feature_dir(selected_entries, base_request)
    proposed_text_feature_cache_path = (
        base_feature_dir / "text" / "text_feature_cache.pt"
        if base_feature_dir is not None
        else Path("outputs") / "features" / dataset / backbone / "text" / "text_feature_cache.pt"
    )
    text_cache_candidates = find_text_feature_cache_candidates(
        entries=selected_entries,
        base_request=base_request,
        proposed_path=proposed_text_feature_cache_path,
        additional_text_dirs=legacy_text_feature_dirs(
            base_feature_dir=base_feature_dir,
            dataset=dataset,
            backbone=backbone,
            base_split_id=base_request["split_id"],
        ),
    )
    candidate_summaries = summarize_text_feature_cache_candidates(
        candidates=text_cache_candidates,
        dataset=dataset,
        backbone=backbone,
        base_split_id=base_request["split_id"],
        class_to_idx=class_to_idx,
        class_names=class_names,
        expected_feature_dim=expected_feature_dim,
        prompt_templates=prompt_templates,
        execution_env=execution_env,
        run_mode=run_mode,
    )
    text_cache_path = choose_text_cache_candidate(candidate_summaries, warnings)
    text_cache_exists = text_cache_path is not None and text_cache_path.exists()
    text_cache_inspection: dict[str, Any] = {
        "path": str(text_cache_path) if text_cache_path is not None else None,
        "exists": text_cache_exists,
        "checked": False,
    }

    if text_cache_exists and text_cache_path is not None:
        text_cache_inspection = inspect_text_feature_cache(
            text_cache_path,
            dataset=dataset,
            backbone=backbone,
            base_split_id=base_request["split_id"],
            class_to_idx=class_to_idx,
            class_names=class_names,
            expected_feature_dim=expected_feature_dim,
            prompt_templates=prompt_templates,
            execution_env=execution_env,
            run_mode=run_mode,
            errors=errors,
            warnings=warnings,
        )
    else:
        recommendations.append(
            "Run a separate text feature extraction script to create a standalone text feature cache; "
            "do not attach text_features to existing train/val/test image feature caches."
        )

    text_feature_cache_ready = bool(text_cache_exists and not errors)
    report = {
        "is_valid": not errors,
        "text_feature_cache_ready": text_feature_cache_ready,
        "text_feature_cache_exists": text_cache_exists,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "recommendations": recommendations,
        "proposed_text_feature_cache_schema": proposed_text_feature_cache_schema(
            num_classes=len(class_names) if class_names else None,
            feature_dim=expected_feature_dim,
            prompt_templates=prompt_templates,
        ),
        "proposed_text_feature_cache_path": str(proposed_text_feature_cache_path),
        "selected_text_feature_cache_path": str(text_cache_path) if text_cache_path is not None else None,
        "text_feature_cache_candidates": candidate_summaries,
        "text_feature_cache_inspection": text_cache_inspection,
        "dataset": dataset,
        "backbone": backbone,
        "seed": infer_seed(base_request),
        "base_split": base_split,
        "checked_base_split": describe_base_request(base_request),
        "class_order_determinable": bool(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names) if class_names else None,
        "expected_feature_dim": expected_feature_dim,
        "prompt_templates": prompt_templates,
        "prompt_template_source": prompt_template_source,
        "manifest_path": str(manifest_source),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "loads_model": False,
        "extracts_text_features": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "saves_predictions": False,
        "writes_results_raw": False,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/check_text_feature_cache_preflight.py",
    }
    destination_dir = unique_dir(output_root / f"{dataset}_{backbone}_{report['seed']}")
    report_path = safe_write_json(destination_dir / "text_feature_cache_preflight_report.json", report)
    return report_path, bool(report["is_valid"])


def resolve_manifest_entries(manifest: dict[str, Any], manifest_path: Path, errors: list[str]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        errors.append("manifest must contain an entries list")
        return []
    manifest_dir = manifest_path.parent
    resolved_entries = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"manifest entry at index {index} must be a mapping")
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


def inspect_class_mapping(split_data: dict[str, Any], errors: list[str]) -> tuple[dict[str, int], list[str]]:
    raw_mapping = split_data.get("class_to_idx")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        errors.append("base split must contain a non-empty class_to_idx mapping")
        return {}, []
    class_to_idx: dict[str, int] = {}
    for name, value in raw_mapping.items():
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append("class_to_idx values must be integer class indices")
            return {}, []
        class_to_idx[str(name)] = int(value)
    sorted_values = sorted(class_to_idx.values())
    if sorted_values != list(range(len(sorted_values))):
        errors.append("class_to_idx values must be contiguous from 0 to num_classes-1")
        return class_to_idx, []
    declared_num_classes = split_data.get("num_classes")
    if isinstance(declared_num_classes, int) and declared_num_classes != len(class_to_idx):
        errors.append(f"base split num_classes={declared_num_classes} does not match class_to_idx size={len(class_to_idx)}")
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    return class_to_idx, class_names


def load_prompt_templates(warnings: list[str], errors: list[str]) -> tuple[list[str], str]:
    config_path = Path("configs") / "methods" / "zero_shot_clip.yaml"
    prompt_templates: list[str] = []
    source = "default"
    if config_path.exists():
        try:
            config = load_yaml_config(config_path)
            method = config.get("method") if isinstance(config.get("method"), dict) else {}
            raw_templates = method.get("prompt_templates") if isinstance(method, dict) else None
            if isinstance(raw_templates, list):
                prompt_templates = [str(item) for item in raw_templates if isinstance(item, str) and item]
                source = str(config_path)
        except ConfigError as exc:
            warnings.append(f"could not load prompt templates from {config_path}: {exc}")
    if not prompt_templates:
        prompt_templates = list(DEFAULT_PROMPT_TEMPLATES)
        source = "built_in_default"
    if not prompt_templates:
        errors.append("prompt templates are not available from config or defaults")
    for template in prompt_templates:
        if "{}" not in template:
            errors.append(f"prompt template must contain '{{}}' class-name placeholder: {template}")
    return prompt_templates, source


def infer_expected_feature_dim(
    *,
    entries: list[dict[str, Any]],
    base_request: dict[str, Any],
    backbone: str,
    warnings: list[str],
) -> int | None:
    dims = []
    for entry in entries:
        if not entry_matches_request(entry, base_request):
            continue
        shape = shape_list(entry.get("feature_shape") or entry.get("image_feature_shape"))
        if len(shape) == 2:
            dims.append(shape[1])
        value = int_or_none(entry.get("feature_dim"))
        if value is not None:
            dims.append(value)
    unique_dims = sorted(set(dims))
    if len(unique_dims) == 1:
        return unique_dims[0]
    if len(unique_dims) > 1:
        warnings.append(f"manifest feature dimensions are inconsistent for base split: {unique_dims}")
        return None

    config_path = Path("configs") / "backbones" / f"{backbone}.yaml"
    if config_path.exists():
        try:
            config = load_yaml_config(config_path)
            backbone_config = config.get("backbone") if isinstance(config.get("backbone"), dict) else {}
            return int_or_none(backbone_config.get("feature_dim"))
        except ConfigError as exc:
            warnings.append(f"could not load feature_dim from {config_path}: {exc}")
    return None


def infer_base_feature_dir(entries: list[dict[str, Any]], base_request: dict[str, Any]) -> Path | None:
    for section in BASE_SECTIONS:
        matches = [
            entry
            for entry in entries
            if entry.get("split_section") == section and entry_matches_request(entry, base_request)
        ]
        if not matches:
            continue
        entry = sorted(matches, key=lambda item: str(item.get("run_dir", item.get("feature_cache_path", ""))))[0]
        run_dir = path_or_none(entry.get("run_dir"))
        if run_dir is not None and run_dir.parent.name == section:
            return run_dir.parent.parent
        cache_path = path_or_none(entry.get("feature_cache_path"))
        if cache_path is not None and cache_path.parent.parent.name == section:
            return cache_path.parent.parent.parent
    return None


def find_text_feature_cache_candidates(
    *,
    entries: list[dict[str, Any]],
    base_request: dict[str, Any],
    proposed_path: Path,
    additional_text_dirs: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if proposed_path.exists():
        candidates.append(proposed_path)
    text_dirs = [proposed_path.parent, *(additional_text_dirs or [])]
    for text_dir in text_dirs:
        if text_dir.exists() and text_dir.is_dir():
            for pattern in ("text_feature_cache.pt", "feature_cache.pt", "*.pt"):
                candidates.extend(sorted(text_dir.rglob(pattern)))
    for entry in entries:
        if not entry_matches_request(entry, base_request) and not path_mentions_text(entry):
            continue
        if not is_text_cache_entry(entry):
            continue
        for key in ("text_feature_cache_path", "feature_cache_path", "cache_path"):
            path = path_or_none(entry.get(key))
            if path is not None and path.exists():
                candidates.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve() if candidate.exists() else candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def legacy_text_feature_dirs(
    *,
    base_feature_dir: Path | None,
    dataset: str,
    backbone: str,
    base_split_id: str,
) -> list[Path]:
    if base_feature_dir is None:
        return []
    candidates = []
    if base_feature_dir.name == backbone and base_feature_dir.parent.name == dataset:
        features_root = base_feature_dir.parent.parent
        candidates.append(features_root / backbone / dataset / base_split_id / dataset / backbone / "text")
    if base_feature_dir.name == base_split_id and base_feature_dir.parent.name == dataset:
        candidates.append(base_feature_dir / dataset / backbone / "text")
    unique = []
    seen = {str((base_feature_dir / "text").resolve())}
    for candidate in candidates:
        key = str(candidate.resolve() if candidate.exists() else candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def summarize_text_feature_cache_candidates(
    *,
    candidates: list[Path],
    dataset: str,
    backbone: str,
    base_split_id: str,
    class_to_idx: dict[str, int],
    class_names: list[str],
    expected_feature_dim: int | None,
    prompt_templates: list[str],
    execution_env: str,
    run_mode: str,
) -> list[dict[str, Any]]:
    summaries = []
    for path in candidates:
        local_errors: list[str] = []
        local_warnings: list[str] = []
        inspection = inspect_text_feature_cache(
            path,
            dataset=dataset,
            backbone=backbone,
            base_split_id=base_split_id,
            class_to_idx=class_to_idx,
            class_names=class_names,
            expected_feature_dim=expected_feature_dim,
            prompt_templates=prompt_templates,
            execution_env=execution_env,
            run_mode=run_mode,
            errors=local_errors,
            warnings=local_warnings,
        )
        shape_valid = bool(
            class_names
            and expected_feature_dim is not None
            and inspection.get("text_feature_shape") == [len(class_names), expected_feature_dim]
        )
        selectable = bool(path.exists() and shape_valid and not local_errors)
        summary = {
            "path": str(path),
            "dry_run": bool(inspection.get("dry_run", False)),
            "uses_fake_text_features": bool(inspection.get("uses_fake_text_features", False)),
            "is_paper_result": inspection.get("is_paper_result"),
            "text_feature_shape": inspection.get("text_feature_shape", []),
            "created_at": inspection.get("created_at"),
            "timestamp": candidate_timestamp(path, inspection.get("created_at")),
            "selectable": selectable,
            "selection_reason": selection_reason(inspection, local_errors, shape_valid),
            "errors": local_errors,
            "warnings": local_warnings,
        }
        summaries.append(summary)
    ranked = sorted(summaries, key=candidate_rank_key, reverse=True)
    for index, summary in enumerate(ranked, start=1):
        summary["selection_rank"] = index
    return ranked


def choose_text_cache_candidate(candidate_summaries: list[dict[str, Any]], warnings: list[str]) -> Path | None:
    if not candidate_summaries:
        return None
    if len(candidate_summaries) > 1:
        warnings.append(
            f"found {len(candidate_summaries)} text feature cache candidates; selecting highest-ranked cache by "
            "real/non-fake status, schema validity, and newest timestamp"
        )
    selected = candidate_summaries[0]
    if selected.get("dry_run") or selected.get("uses_fake_text_features"):
        warnings.append(
            "selected text feature cache uses fake/dry-run text features; it is acceptable for preflight shape checks "
            "but must not be used for real zero-shot evaluation"
        )
    return Path(str(selected["path"]))


def candidate_rank_key(summary: dict[str, Any]) -> tuple[int, int, int, int, int, str, str]:
    return (
        int(bool(summary.get("selectable"))),
        int(not bool(summary.get("dry_run"))),
        int(not bool(summary.get("uses_fake_text_features"))),
        int(summary.get("is_paper_result") is False),
        int(bool(summary.get("text_feature_shape"))),
        str(summary.get("timestamp") or ""),
        str(summary.get("path") or ""),
    )


def selection_reason(inspection: dict[str, Any], errors: list[str], shape_valid: bool) -> str:
    if errors:
        return "not selectable: " + "; ".join(errors)
    if not shape_valid:
        return "not selectable: text_features shape does not match expected [num_classes, feature_dim]"
    if bool(inspection.get("dry_run")) or bool(inspection.get("uses_fake_text_features")):
        return "selectable for preflight only: dry-run/fake text features"
    return "selectable real standalone text feature cache"


def candidate_timestamp(path: Path, created_at: Any) -> str:
    if isinstance(created_at, str) and created_at:
        return created_at
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{8}T\d{6}(?:_\d+)?", part):
            return part
    return ""


def inspect_text_feature_cache(
    path: Path,
    *,
    dataset: str,
    backbone: str,
    base_split_id: str,
    class_to_idx: dict[str, int],
    class_names: list[str],
    expected_feature_dim: int | None,
    prompt_templates: list[str],
    execution_env: str,
    run_mode: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    inspection: dict[str, Any] = {"path": str(path), "exists": True, "checked": True}
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        errors.append(f"failed to read text feature cache {path}: {exc}")
        inspection["read_error"] = str(exc)
        return inspection
    if not isinstance(data, dict):
        errors.append(f"{path}: text feature cache must contain a mapping")
        inspection["is_mapping"] = False
        return inspection

    text_shape = list(shape_of_2d(data.get("text_features")))
    prompts = prompt_values(data)
    cache_class_to_idx = data.get("class_to_idx")
    cache_class_names = [str(item) for item in data.get("class_names", [])] if isinstance(data.get("class_names"), list) else []
    feature_dim = int_or_none(data.get("feature_dim")) or (text_shape[1] if len(text_shape) == 2 else None)
    num_classes = int_or_none(data.get("num_classes")) or (text_shape[0] if len(text_shape) == 2 else None)
    inspection.update(
        {
            "text_feature_shape": text_shape,
            "dataset": data.get("dataset"),
            "backbone": data.get("backbone"),
            "base_split": data.get("base_split"),
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "class_names": cache_class_names,
            "has_class_to_idx": isinstance(cache_class_to_idx, dict) and bool(cache_class_to_idx),
            "num_prompts": len(prompts),
            "normalize_features": data.get("normalize_features"),
            "source_script": data.get("source_script"),
            "created_at": data.get("created_at"),
            "git_commit": data.get("git_commit"),
            "execution_env": data.get("execution_env"),
            "run_mode": data.get("run_mode"),
            "dry_run": data.get("dry_run"),
            "uses_fake_text_features": data.get("uses_fake_text_features"),
            "is_paper_result": data.get("is_paper_result"),
        }
    )

    expected_shape = [len(class_names), expected_feature_dim] if class_names and expected_feature_dim is not None else None
    if expected_shape is not None and text_shape != expected_shape:
        errors.append(f"{path}: text_features shape {text_shape} does not equal expected {expected_shape}")
    if data.get("dataset") != dataset:
        errors.append(f"{path}: dataset mismatch, expected {dataset}, found {data.get('dataset')}")
    if data.get("backbone") != backbone:
        errors.append(f"{path}: backbone mismatch, expected {backbone}, found {data.get('backbone')}")
    if not data.get("base_split"):
        errors.append(f"{path}: base_split must be recorded")
    elif str(data.get("base_split", "")) not in {base_split_id, str(Path(base_split_id).stem)}:
        warnings.append(f"{path}: base_split is not recorded as {base_split_id}")
    if feature_dim is not None and expected_feature_dim is not None and feature_dim != expected_feature_dim:
        errors.append(f"{path}: feature_dim={feature_dim} does not match expected_feature_dim={expected_feature_dim}")
    if num_classes is not None and class_names and num_classes != len(class_names):
        errors.append(f"{path}: num_classes={num_classes} does not match base split class count={len(class_names)}")
    if data.get("feature_dim") is None:
        errors.append(f"{path}: feature_dim must be recorded")
    if data.get("num_classes") is None:
        errors.append(f"{path}: num_classes must be recorded")
    if cache_class_to_idx != class_to_idx:
        errors.append(f"{path}: class_to_idx does not match base split")
    if not cache_class_names:
        errors.append(f"{path}: class_names must be recorded")
    if cache_class_names and cache_class_names != class_names:
        errors.append(f"{path}: class_names do not match class_to_idx order from base split")
    if not prompts:
        errors.append(f"{path}: prompts or prompt_templates must be stored")
    elif prompt_templates and prompts != prompt_templates and len(prompts) < len(class_names):
        warnings.append(f"{path}: stored prompt metadata differs from zero-shot default templates")
    if data.get("normalize_features") is None:
        errors.append(f"{path}: normalize_features must be recorded")
    if not data.get("source_script"):
        errors.append(f"{path}: source_script must be recorded")
    if not data.get("created_at"):
        errors.append(f"{path}: created_at must be recorded")
    if not data.get("git_commit"):
        errors.append(f"{path}: git_commit must be recorded")
    if not data.get("execution_env"):
        errors.append(f"{path}: execution_env must be recorded")
    elif data.get("execution_env") != execution_env:
        warnings.append(f"{path}: execution_env differs from this preflight request")
    if not data.get("run_mode"):
        errors.append(f"{path}: run_mode must be recorded")
    elif data.get("run_mode") != run_mode:
        warnings.append(f"{path}: run_mode differs from this preflight request")
    if bool(data.get("dry_run")) or bool(data.get("uses_fake_text_features")):
        warnings.append(f"{path}: dry_run/uses_fake_text_features cache is not suitable for real zero-shot evaluation")
    if data.get("is_paper_result") is not False:
        errors.append(f"{path}: is_paper_result must be false for text feature cache preflight readiness")
    return inspection


def proposed_text_feature_cache_schema(
    *,
    num_classes: int | None,
    feature_dim: int | None,
    prompt_templates: list[str],
) -> dict[str, Any]:
    return {
        "text_features": {"required": True, "shape": [num_classes or "num_classes", feature_dim or "feature_dim"]},
        "class_names": {"required": True, "order": "class_to_idx sorted by integer index"},
        "class_to_idx": {"required": True},
        "prompts_or_prompt_templates": {"required": True, "prompt_templates": prompt_templates},
        "dataset": {"required": True},
        "backbone": {"required": True},
        "base_split": {"required": True},
        "feature_dim": {"required": True},
        "num_classes": {"required": True},
        "normalize_features": {"required": True},
        "source_script": {"required": True},
        "created_at": {"required": True},
        "git_commit": {"required": True},
        "execution_env": {"required": True},
        "run_mode": {"required": True},
        "is_paper_result": {"required": True, "value": False},
    }


def entry_matches_request(entry: dict[str, Any], request: dict[str, Any]) -> bool:
    tokens = request["tokens"]
    for key in ("summary_path", "run_dir", "feature_cache_path", "split_path", "text_feature_cache_path"):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if tokens & {value, path.name, path.stem, *path.parts}:
            return True
    return False


def is_text_cache_entry(entry: dict[str, Any]) -> bool:
    section = str(entry.get("split_section", "")).lower()
    if section in TEXT_SECTION_NAMES:
        return True
    return path_mentions_text(entry)


def path_mentions_text(entry: dict[str, Any]) -> bool:
    for key in ("summary_path", "run_dir", "feature_cache_path", "text_feature_cache_path"):
        value = entry.get(key)
        if isinstance(value, str) and any("text" in part.lower() for part in Path(value).parts):
            return True
    return False


def prompt_values(data: dict[str, Any]) -> list[str]:
    for key in ("prompts", "prompt_templates", "text_prompts"):
        value = data.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def shape_list(value: Any) -> list[int]:
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
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def resolve_cache_path(entry: dict[str, Any], base_dir: Path) -> Path | None:
    path = resolve_path(entry.get("feature_cache_path"), base_dir)
    if path is not None:
        return path
    path = resolve_path(entry.get("text_feature_cache_path"), base_dir)
    if path is not None:
        return path
    run_dir = resolve_path(entry.get("run_dir"), base_dir)
    if run_dir is not None:
        text_candidate = run_dir / "text_feature_cache.pt"
        if text_candidate.exists():
            return text_candidate
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


def path_or_none(value: Any) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def infer_seed(request: dict[str, Any]) -> str:
    split_data = request.get("split_data")
    if isinstance(split_data, dict) and split_data.get("seed") is not None:
        return f"seed{split_data['seed']}"
    match = re.search(r"seed(\d+)", request["split_id"])
    if match:
        return f"seed{match.group(1)}"
    return "seed_unknown"


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
            raise ValueError("text feature cache preflight reports must not be written under results/raw")


if __name__ == "__main__":
    main()
