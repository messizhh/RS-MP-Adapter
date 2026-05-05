#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


DEFAULT_SECTIONS = ["train", "val", "test", "support"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely verify image files referenced by a split JSON.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sections", nargs="+", default=DEFAULT_SECTIONS, choices=DEFAULT_SECTIONS)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="local_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, num_failed = run_image_open_preflight(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        split_path=args.split,
        sections=args.sections,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        source_script="scripts/check_image_open_preflight.py",
    )
    print(f"image_open_report_path={report_path}")
    if num_failed:
        raise SystemExit(1)


def run_image_open_preflight(
    *,
    dataset: str,
    dataset_root: str | Path,
    split_path: str | Path,
    sections: list[str],
    max_samples: int,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    source_script: str,
) -> tuple[Path, int]:
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative")

    split = read_json(split_path)
    root = Path(dataset_root)
    failures: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    num_checked = 0

    for section in sections:
        samples = split.get(section, [])
        if not isinstance(samples, list):
            failures.append({"section": section, "error": "section is not a list"})
            continue
        for sample_index, sample in enumerate(samples[:max_samples]):
            num_checked += 1
            if not isinstance(sample, dict):
                failures.append({"section": section, "sample_index": sample_index, "error": "sample is not an object"})
                continue
            summary, failure = check_sample_image(root, section, sample_index, sample)
            if failure is not None:
                failures.append(failure)
            if summary is not None:
                image_summaries.append(summary)

    report = {
        "dataset": dataset,
        "dataset_root": str(root),
        "split_path": str(split_path),
        "checked_sections": sections,
        "max_samples": max_samples,
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
    run_dir = unique_dir(Path(output_dir) / dataset / "image_open_preflight")
    report_path = safe_write_json(run_dir / "image_open_preflight_report.json", report, overwrite=False)
    return report_path, len(failures)


def check_sample_image(
    dataset_root: Path,
    section: str,
    sample_index: int,
    sample: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sample_path_value = sample.get("path")
    failure_base = {
        "section": section,
        "sample_index": sample_index,
        "sample_path": sample_path_value,
        "class_name": sample.get("class_name"),
        "label": sample.get("label"),
    }
    if not isinstance(sample_path_value, str) or not sample_path_value:
        return None, {**failure_base, "error": "sample path is missing or not a string"}

    sample_path = Path(sample_path_value)
    if sample_path.is_absolute():
        return None, {**failure_base, "error": "absolute sample paths are not allowed"}

    image_path = dataset_root / sample_path
    try:
        resolved_root = dataset_root.resolve(strict=False)
        resolved_image_path = image_path.resolve(strict=False)
        if not resolved_image_path.is_relative_to(resolved_root):
            return None, {**failure_base, "resolved_path": str(resolved_image_path), "error": "sample path escapes dataset root"}
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            original_format = image.format
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            mode = rgb_image.mode
    except Exception as exc:
        return None, {**failure_base, "resolved_path": str(image_path), "error": str(exc)}

    summary = {
        "section": section,
        "sample_index": sample_index,
        "sample_path": sample_path_value,
        "resolved_path": str(image_path),
        "class_name": sample.get("class_name"),
        "label": sample.get("label"),
        "width": width,
        "height": height,
        "mode": mode,
        "format": original_format,
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
    raise FileExistsError(f"Could not create unique image-open preflight directory under {base}")


if __name__ == "__main__":
    main()
