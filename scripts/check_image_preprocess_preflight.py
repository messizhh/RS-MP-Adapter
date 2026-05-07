#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.features.image_loader import ImageSample, load_split_samples
from src.features.image_preprocess import inspect_image_metadata
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


DEFAULT_SECTIONS = ["train", "val", "test", "support"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely verify PIL RGB/resize preprocessing for split images.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--sections", nargs="+", default=DEFAULT_SECTIONS, choices=DEFAULT_SECTIONS)
    parser.add_argument("--max-samples-per-section", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, num_failed = run_image_preprocess_preflight(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        split_path=args.split,
        backbone_config_path=args.backbone_config,
        sections=args.sections,
        max_samples_per_section=args.max_samples_per_section,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        source_script="scripts/check_image_preprocess_preflight.py",
    )
    print(f"image_preprocess_report_path={report_path}")
    if num_failed:
        raise SystemExit(1)


def run_image_preprocess_preflight(
    *,
    dataset: str,
    dataset_root: str | Path,
    split_path: str | Path,
    backbone_config_path: str | Path,
    sections: list[str],
    max_samples_per_section: int,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    source_script: str,
) -> tuple[Path, int]:
    if max_samples_per_section < 0:
        raise ValueError("--max-samples-per-section must be non-negative")

    split = read_json(split_path)
    image_size = image_size_from_backbone_config(backbone_config_path)
    failures: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    num_checked = 0

    for section in sections:
        section_samples = split.get(section, [])
        if not isinstance(section_samples, list):
            failures.append({"section": section, "error": "section is not a list"})
            continue
        num_checked += min(len(section_samples), max_samples_per_section)
        try:
            samples = load_split_samples(
                split_path=split_path,
                dataset_root=dataset_root,
                sections=[section],
                max_samples=max_samples_per_section,
            )
        except Exception as exc:
            failures.append({"section": section, "error": str(exc)})
            continue

        for sample in samples:
            summary, failure = inspect_sample(sample, image_size)
            if failure is not None:
                failures.append(failure)
            if summary is not None:
                image_summaries.append(summary)

    report = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "split_path": str(split_path),
        "backbone_config": str(backbone_config_path),
        "image_size": image_size,
        "checked_sections": sections,
        "max_samples_per_section": max_samples_per_section,
        "num_checked": num_checked,
        "num_failed": len(failures),
        "failures": failures,
        "image_summaries": image_summaries,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "reads_image_pixels": True,
        "loads_model": False,
        "extracts_features": False,
        "trains_model": False,
        "evaluates_model": False,
        "source_script": source_script,
        "created_at": utc_now_iso(),
    }
    run_dir = unique_dir(Path(output_dir) / dataset / "image_preprocess_preflight")
    report_path = safe_write_json(run_dir / "image_preprocess_preflight_report.json", report, overwrite=False)
    return report_path, len(failures)


def image_size_from_backbone_config(backbone_config_path: str | Path) -> int:
    config = load_yaml_config(backbone_config_path)
    backbone = config.get("backbone", {})
    if not isinstance(backbone, dict):
        return 224
    preprocess = backbone.get("preprocess", {})
    if isinstance(preprocess, dict) and isinstance(preprocess.get("resize"), int):
        return preprocess["resize"]
    if isinstance(backbone.get("image_size"), int):
        return backbone["image_size"]
    return 224


def inspect_sample(sample: ImageSample, image_size: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    failure_base = {
        "section": sample.section,
        "sample_index": sample.sample_index,
        "sample_path": sample.relative_path,
        "resolved_path": str(sample.image_path),
        "class_name": sample.class_name,
        "label": sample.label,
    }
    try:
        metadata = inspect_image_metadata(sample.image_path, image_size=image_size)
    except Exception as exc:
        return None, {**failure_base, "error": str(exc)}

    summary = {
        **failure_base,
        "width": metadata["width"],
        "height": metadata["height"],
        "mode": metadata["mode"],
        "reads_image_pixels": metadata["reads_image_pixels"],
        "loads_model": metadata["loads_model"],
        "extracts_features": metadata["extracts_features"],
        "trains_model": metadata["trains_model"],
        "evaluates_model": metadata["evaluates_model"],
        "is_paper_result": metadata["is_paper_result"],
    }
    return summary, None


def unique_dir(base: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique image preprocess preflight directory under {base}")


if __name__ == "__main__":
    main()
