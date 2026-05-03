#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config_loader import load_yaml_config
from src.features.extract_features import run_dry_run_feature_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract or dry-run feature caches.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default="extract_features")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--feature-cache", default="", help="Reserved for CLI consistency; extraction writes a new cache.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone_config = load_yaml_config(args.config) if args.config else {"backbone": {"name": args.backbone, "family": "fake", "feature_dim": 8}}
    if not args.dry_run:
        raise SystemExit("Real feature extraction is not implemented in Phase 1D. Use --dry-run for local shape validation.")
    if args.execution_env == "local_wsl" and args.device != "cpu":
        raise SystemExit("Local WSL dry-run feature extraction must use --device cpu.")
    result = run_dry_run_feature_extraction(
        dataset=args.dataset,
        backbone_name=args.backbone,
        backbone_config=backbone_config,
        output_dir=args.output_dir,
        split_path=args.split,
        max_samples=args.max_samples,
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


if __name__ == "__main__":
    main()
