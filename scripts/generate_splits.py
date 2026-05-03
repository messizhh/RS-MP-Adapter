#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.datasets.base_dataset import descriptor_from_config
from src.datasets.split_generator import DEFAULT_SEEDS, DEFAULT_SHOTS, generate_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed train/val/test and few-shot splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="")
    parser.add_argument("--method", default="generate_splits")
    parser.add_argument("--shot", type=int, default=None, help="Reserved for CLI consistency; use --shots to write split files.")
    parser.add_argument("--seed", type=int, default=None, help="Reserved for CLI consistency; use --seeds to write split files.")
    parser.add_argument("--split", default="", help="Reserved for CLI consistency; split generation writes new splits.")
    parser.add_argument("--feature-cache", default="", help="Reserved for CLI consistency; split generation does not read feature caches.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing split JSON files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--min-samples-per-class", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples_per_class is None:
        args.max_samples_per_class = args.max_samples
    if args.execution_env == "local_wsl" and args.device != "cpu":
        raise SystemExit("Local WSL split generation must use --device cpu.")
    config = load_yaml_config(args.config)
    dataset_cfg = config.get("dataset", {})
    root = args.dataset_root or dataset_cfg.get("root")
    if root is None:
        raise SystemExit("Dataset root must be provided by config dataset.root or --dataset-root")
    output_dir = args.output_dir or dataset_cfg.get("output_split_root")
    if output_dir is None:
        raise SystemExit("Output directory must be provided by config dataset.output_split_root or --output-dir")
    split_cfg = dataset_cfg.get("split", {})
    descriptor = descriptor_from_config(config, dataset_name=args.dataset, dataset_root=root)
    min_samples = args.min_samples_per_class
    if min_samples is None:
        min_samples = int(dataset_cfg.get("min_images_per_class", 1))
    written = generate_split_files(
        dataset=args.dataset,
        root=root,
        output_dir=output_dir,
        shots=args.shots,
        seeds=args.seeds,
        train_ratio=float(split_cfg.get("train_ratio", 0.6)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        source_script="scripts/generate_splits.py",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_samples_per_class=args.max_samples_per_class,
        min_samples_per_class=min_samples,
        descriptor=descriptor,
        split_policy=str(split_cfg.get("policy", "class_stratified_random")),
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        is_paper_result=False,
    )
    prefix = "dry_run_path" if args.dry_run else "written_path"
    for path in written:
        print(f"{prefix}={Path(path)}")


if __name__ == "__main__":
    main()
