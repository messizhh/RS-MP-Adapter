#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.datasets.base_dataset import descriptor_from_config, discover_class_dirs, find_class_root, inspect_class_folder_dataset
from src.datasets.split_generator import DEFAULT_SHOTS
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


VALID_CLASS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only dataset directory preflight for class-folder datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="outputs/preflight")
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    report = check_dataset_layout(
        config=config,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        shots=args.shots,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
    )
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = safe_write_json(unique_path(output_dir / f"dataset_layout_preflight_{timestamp_for_path()}.json"), report)
    print(f"preflight_report_path={report_path}")
    print(f"is_ready_for_split_generation={str(report['is_ready_for_split_generation']).lower()}")
    if not report["is_ready_for_split_generation"]:
        raise SystemExit(2)


def check_dataset_layout(
    config: dict[str, Any],
    dataset: str,
    dataset_root: str | Path,
    shots: list[int],
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
    execution_env: str = "local_wsl",
    run_mode: str = "local_validation",
) -> dict[str, Any]:
    descriptor = descriptor_from_config(config, dataset_name=dataset, dataset_root=dataset_root)
    split_cfg = (config.get("dataset", {}) or {}).get("split", {}) or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.6))
    warnings: list[str] = []
    errors: list[str] = []
    class_root = None
    root = Path(dataset_root)

    base_report: dict[str, Any] = {
        "dataset": dataset,
        "display_name": descriptor.display_name,
        "dataset_root": str(root),
        "root_exists": root.exists(),
        "root_is_dir": root.is_dir() if root.exists() else False,
        "class_root": "",
        "expected_num_classes": descriptor.expected_num_classes,
        "num_classes": 0,
        "num_images": 0,
        "class_counts": {},
        "empty_classes": [],
        "non_image_files": {},
        "duplicate_class_names": [],
        "invalid_class_names": [],
        "supports_shots": {str(shot): False for shot in shots},
        "shot_failures": {str(shot): [] for shot in shots},
        "warnings": warnings,
        "errors": errors,
        "image_extensions": list(descriptor.image_extensions),
        "class_folder_candidates": list(descriptor.class_folder_candidates),
        "train_ratio_for_support_check": train_ratio,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "created_at": utc_now_iso(),
        "source_script": "scripts/check_dataset_layout.py",
    }

    if not root.exists():
        errors.append(f"Dataset root does not exist: {root}")
        return finalize_report(base_report)
    if not root.is_dir():
        errors.append(f"Dataset root is not a directory: {root}")
        return finalize_report(base_report)

    try:
        class_root = find_class_root(descriptor)
        base_report["class_root"] = str(class_root)
    except Exception as exc:
        errors.append(str(exc))
        return finalize_report(base_report)

    class_dirs = discover_class_dirs(class_root, descriptor.ignore_hidden_files)
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]
        warnings.append(f"Preflight limited to max_classes={max_classes}; do not use this report for final split readiness.")
    class_names = [path.name for path in class_dirs]
    base_report["duplicate_class_names"] = find_duplicate_class_names(class_names)
    base_report["invalid_class_names"] = find_invalid_class_names(class_names)
    if base_report["duplicate_class_names"]:
        errors.append("Duplicate class names found after normalization.")
    if base_report["invalid_class_names"]:
        errors.append("Invalid class names found; class names should be stable folder names.")

    try:
        inspection = inspect_class_folder_dataset(
            descriptor,
            max_classes=max_classes,
            max_samples_per_class=max_samples_per_class,
        )
    except Exception as exc:
        errors.append(str(exc))
        return finalize_report(base_report)

    class_counts = {row["class_name"]: int(row["num_supported_images"]) for row in inspection["class_summary"]}
    empty_classes = [name for name, count in class_counts.items() if count == 0]
    non_image_files = dict(inspection.get("unsupported_extensions", {}))
    if non_image_files:
        warnings.append("Non-image or unsupported files were found and will be ignored by split generation.")
    if empty_classes:
        errors.append("One or more class folders contain no supported images.")
    for critical_error in inspection.get("critical_errors", []):
        if critical_error not in errors:
            errors.append(critical_error)

    supports_shots, shot_failures = compute_shot_support(class_counts, shots, train_ratio)
    unsupported_shots = [shot for shot, supported in supports_shots.items() if not supported]
    if unsupported_shots:
        warnings.append(f"Dataset does not support all requested shots from the configured train split: {', '.join(unsupported_shots)}")

    base_report.update(
        {
            "num_classes": int(inspection["num_classes"]),
            "num_images": int(inspection["num_samples"]),
            "class_counts": class_counts,
            "empty_classes": empty_classes,
            "non_image_files": non_image_files,
            "supports_shots": supports_shots,
            "shot_failures": shot_failures,
        }
    )
    return finalize_report(base_report)


def compute_shot_support(class_counts: dict[str, int], shots: list[int], train_ratio: float) -> tuple[dict[str, bool], dict[str, list[str]]]:
    supports: dict[str, bool] = {}
    failures: dict[str, list[str]] = {}
    for shot in shots:
        failed_classes = []
        for class_name, count in class_counts.items():
            estimated_train = max(1, int(count * train_ratio)) if count > 0 else 0
            if estimated_train < shot:
                failed_classes.append(class_name)
        supports[str(shot)] = not failed_classes and bool(class_counts)
        failures[str(shot)] = failed_classes
    return supports, failures


def find_duplicate_class_names(class_names: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    duplicates = []
    for name in class_names:
        normalized = name.strip().casefold()
        if normalized in seen:
            duplicates.append(name)
        else:
            seen[normalized] = name
    return sorted(duplicates)


def find_invalid_class_names(class_names: list[str]) -> list[str]:
    invalid = []
    for name in class_names:
        if name != name.strip() or not VALID_CLASS_NAME.match(name):
            invalid.append(name)
    return sorted(invalid)


def finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["has_non_image_files"] = bool(report["non_image_files"])
    report["has_empty_classes"] = bool(report["empty_classes"])
    report["is_ready_for_split_generation"] = (
        report["root_exists"]
        and report["root_is_dir"]
        and not report["errors"]
        and all(bool(value) for value in report["supports_shots"].values())
    )
    return report


def timestamp_for_path() -> str:
    return utc_now_iso().replace(":", "").replace("-", "").split(".")[0]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find non-existing output path for {path}")


if __name__ == "__main__":
    main()
