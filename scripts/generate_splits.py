#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.datasets.split_generator import DEFAULT_SEEDS, DEFAULT_SHOTS, generate_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed train/val/test and few-shot splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing split JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    dataset_cfg = config.get("dataset", {})
    root = args.dataset_root or dataset_cfg.get("root")
    if root is None:
        raise SystemExit("Dataset root must be provided by config dataset.root or --dataset-root")
    split_cfg = dataset_cfg.get("split", {})
    written = generate_split_files(
        dataset=args.dataset,
        root=root,
        output_dir=args.output_dir,
        shots=args.shots,
        seeds=args.seeds,
        train_ratio=float(split_cfg.get("train_ratio", 0.6)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        source_script="scripts/generate_splits.py",
        overwrite=args.overwrite,
    )
    for path in written:
        print(Path(path))


if __name__ == "__main__":
    main()
