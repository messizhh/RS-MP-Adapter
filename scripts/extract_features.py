#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.features.extract_features import run_dry_run_feature_extraction, run_guarded_real_feature_extraction, shell_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract or dry-run feature caches.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-section", default="test", choices=["support", "train", "val", "test"])
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--method", default="extract_features")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--feature-cache", default="", help="Reserved for CLI consistency; extraction writes a new cache.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-real-extraction", action="store_true")
    parser.add_argument("--paper-result-candidate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone_config = load_yaml_config(args.config) if args.config else {"backbone": {"name": args.backbone, "family": "fake", "feature_dim": 8}}
    if not args.dry_run:
        if not args.allow_real_extraction:
            raise SystemExit("Real feature extraction is guarded. Pass --allow-real-extraction with remote_server server_full/server_benchmark inputs.")
        dataset_root = args.dataset_root or dataset_root_from_config(args.dataset_config)
        if not args.weights_path:
            raise SystemExit("Real feature extraction requires an explicit --weights-path.")
        if not args.split:
            raise SystemExit("Real feature extraction requires an explicit --split path.")
        if not dataset_root:
            raise SystemExit("Real feature extraction requires --dataset-root or --dataset-config with a root/data_root field.")
        result = run_guarded_real_feature_extraction(
            dataset=args.dataset,
            backbone_name=args.backbone,
            backbone_config=backbone_config,
            output_dir=args.output_dir,
            split_path=args.split,
            split_section=args.split_section,
            seed=args.seed,
            shot=args.shot,
            dataset_root=dataset_root,
            weights_path=args.weights_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
            execution_env=args.execution_env,
            run_mode=args.run_mode,
            command=shell_join(sys.argv),
            is_paper_result_candidate=args.paper_result_candidate,
            source_script="scripts/extract_features.py",
        )
        print(f"feature_cache_path={result['cache_path']}")
        print(f"summary_path={result['summary_path']}")
        print(f"run_dir={result['run_dir']}")
        return
    if args.execution_env == "local_wsl" and args.device != "cpu":
        raise SystemExit("Local WSL dry-run feature extraction must use --device cpu.")
    result = run_dry_run_feature_extraction(
        dataset=args.dataset,
        backbone_name=args.backbone,
        backbone_config=backbone_config,
        output_dir=args.output_dir,
        split_path=args.split,
        max_samples=args.max_samples if args.max_samples is not None else 12,
        batch_size=args.batch_size,
        device=args.device,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        prompt_templates=(backbone_config.get("method", {}) or {}).get("prompt_templates"),
        overwrite=args.overwrite,
        source_script="scripts/extract_features.py",
    )
    print(f"feature_cache_path={result['cache_path']}")
    print(f"summary_path={result['summary_path']}")
    print(f"run_dir={result['run_dir']}")


def dataset_root_from_config(config_path: str | None) -> str | None:
    if not config_path:
        return None
    config = load_yaml_config(config_path)
    dataset = config.get("dataset", config)
    if not isinstance(dataset, dict):
        return None
    for key in ("root", "data_root", "dataset_root"):
        value = dataset.get(key)
        if isinstance(value, str) and value:
            return value
    return None


if __name__ == "__main__":
    main()
