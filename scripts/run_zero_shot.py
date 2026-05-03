#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.zero_shot import ZeroShotClassifier
from src.config.config_loader import load_configs, validate_required_fields
from src.features.feature_cache import load_feature_cache, make_fake_feature_cache
from src.logging.experiment_logger import finish_experiment_run, start_experiment_run
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot interface.")
    parser.add_argument("--config", default="configs/methods/zero_shot_clip.yaml")
    parser.add_argument("--env-config", default="configs/env/local_wsl.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default="zero_shot_clip")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="")
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution-env", default=None)
    parser.add_argument("--run-mode", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configs([args.env_config, args.config], args.override)
    validate_required_fields(config, ["execution_env", "run_mode", "device", "is_paper_result", "method.name"])
    execution_env = args.execution_env or config["execution_env"]
    run_mode = args.run_mode or config["run_mode"]
    device = args.device or config["device"]
    set_seed(args.seed, deterministic=True)

    if args.dry_run:
        cache = make_fake_feature_cache(num_samples=args.max_samples or 12, seed=args.seed)
    elif args.feature_cache:
        cache = load_feature_cache(args.feature_cache)
    else:
        raise SystemExit("Provide --feature-cache for real evaluation or use --dry-run for fake-feature smoke validation.")

    run, metadata = start_experiment_run(
        output_dir=args.output_dir,
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        backbone=args.backbone,
        method=args.method,
        shot=args.shot,
        seed=args.seed,
        execution_env=execution_env,
        run_mode=run_mode,
        device=device,
        split_path=args.split,
        is_paper_result=bool(config.get("is_paper_result", False)),
    )
    result = ZeroShotClassifier().evaluate(cache)
    metrics = {
        "method": args.method,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "device": device,
        "top1_acc": result.top1_acc,
        "num_samples": result.num_samples,
        "num_classes": result.num_classes,
        "cache_entries": 0,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "inference_time_sec": result.inference_time_sec,
        "images_per_second": result.images_per_second,
        "gpu_memory_mb": None,
        "uses_fake_data": bool(args.dry_run),
        "uses_fake_features": result.used_fake_features,
        "fake_or_dry_run": bool(args.dry_run or result.used_fake_features),
        "used_fake_features": result.used_fake_features,
        "is_real_evaluation": not result.used_fake_features,
        "config_path": str(Path(args.config)),
        "config_snapshot_path": str(run.config_snapshot_path),
        "split_path": args.split,
        "checkpoint_path": None,
        "prediction_path": None,
        "log_path": str(run.log_path),
    }
    metadata_path, metrics_path = finish_experiment_run(run, metadata, metrics)
    print(f"metadata_path={metadata_path}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
