#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.datasets.base_dataset import descriptor_from_config, inspect_class_folder_dataset
from src.utils.io import safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a class-folder remote sensing dataset root.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="")
    parser.add_argument("--method", default="inspect_dataset")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="")
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples_per_class is None:
        args.max_samples_per_class = args.max_samples
    if args.execution_env == "local_wsl" and args.device != "cpu":
        raise SystemExit("Local WSL dataset inspection must use --device cpu.")
    config = load_yaml_config(args.config) if args.config else {"dataset": {"name": args.dataset, "root": args.dataset_root}}
    descriptor = descriptor_from_config(config, dataset_name=args.dataset, dataset_root=args.dataset_root)
    report = inspect_class_folder_dataset(
        descriptor,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
    )
    report.pop("samples", None)
    report.update(
        {
            "execution_env": args.execution_env,
            "run_mode": args.run_mode,
            "is_paper_result": False,
            "created_at": utc_now_iso(),
            "source_script": "scripts/inspect_dataset.py",
        }
    )
    if not report["is_valid"]:
        report["message"] = "Dataset inspection found critical validation errors."
    else:
        report["message"] = "Dataset inspection passed configured critical checks."

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.dataset}_inspection.json"
    summary_path = output_dir / f"{args.dataset}_class_summary.csv"
    if args.write_report:
        report_path = safe_write_json(unique_path(report_path), report)
        summary_path = safe_write_csv(
            unique_path(summary_path),
            report["class_summary"],
            ["class_name", "class_idx", "class_dir", "num_supported_images", "num_used_images", "is_empty"],
        )
        print(f"report_path={report_path}")
        print(f"class_summary_path={summary_path}")
    print(f"is_valid={str(report['is_valid']).lower()}")
    print(f"num_classes={report['num_classes']}")
    print(f"num_samples={report['num_samples']}")
    if not report["is_valid"]:
        raise SystemExit(2)


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
