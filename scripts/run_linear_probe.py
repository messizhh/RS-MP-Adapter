#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.linear_probe import LinearProbe
from src.baselines.runner_utils import load_or_make_cache, run_training_free_method
from src.config.config_loader import load_configs
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPU-safe linear probe skeleton on cached/fake features.")
    add_common_args(parser)
    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument("--config", default="configs/methods/linear_probe.yaml")
    parser.add_argument("--env-config", default="configs/env/local_wsl.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default="linear_probe")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="")
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")


def main():
    args = parse_args()
    if args.execution_env == "local_wsl" and args.device != "cpu":
        raise SystemExit("Local WSL runs must use --device cpu.")
    set_seed(args.seed, deterministic=True)
    config = load_configs([args.env_config, args.config])
    result = run_training_free_method(args, config, LinearProbe(), load_or_make_cache(args))
    print_paths(result)


def print_paths(result):
    print(f"metadata_path={result['metadata_path']}")
    print(f"metrics_path={result['metrics_path']}")
    print(f"prediction_path={result['prediction_path']}")


if __name__ == "__main__":
    main()
