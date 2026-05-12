#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cached_adapter_runner import add_cached_adapter_args, run_cached_training_free_adapter_evaluation
from src.config.config_loader import load_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Run training-free Proto-Adapter on cached image/text feature caches.")
    add_cached_adapter_args(parser, default_config="configs/methods/proto_adapter.yaml", default_method="proto_adapter")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--finetune", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.finetune:
        raise SystemExit("Proto-Adapter-F fine-tuning is disabled for this cached training-free runner.")
    config_paths = [path for path in [args.env_config, args.config] if path]
    config = load_configs(config_paths, args.override)
    result = run_cached_proto_adapter_evaluation(
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        backbone=args.backbone,
        shot=args.shot,
        seed=args.seed,
        manifest_path=args.manifest,
        base_split=args.base_split,
        shot_split=args.shot_split or args.split,
        text_feature_cache_path=args.text_feature_cache,
        adapter_input_plan=args.adapter_input_plan,
        eval_splits=args.eval_splits,
        output_dir=args.output_dir,
        device=args.device or config.get("device", "cpu"),
        execution_env=args.execution_env or config.get("execution_env", "local_wsl"),
        run_mode=args.run_mode or config.get("run_mode", "local_validation"),
        preflight_report=args.preflight_report,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        alpha=args.alpha,
        temperature=args.temperature,
        save_predictions=args.save_predictions,
        allow_paper_result=args.allow_paper_result,
        command=shlex.join(sys.argv),
    )
    print_paths(result)


def run_cached_proto_adapter_evaluation(**kwargs):
    return run_cached_training_free_adapter_evaluation(method_name="proto_adapter", **kwargs)


def print_paths(result):
    print(f"run_dir={result['run_dir']}")
    print(f"metadata_path={result['metadata_path']}")
    print(f"metrics_path={result['metrics_path']}")
    if result.get("prediction_path"):
        print(f"prediction_path={result['prediction_path']}")


if __name__ == "__main__":
    main()
