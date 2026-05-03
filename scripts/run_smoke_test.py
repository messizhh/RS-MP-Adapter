#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.zero_shot import ZeroShotClassifier
from src.config.config_loader import load_configs, validate_required_fields
from src.datasets.base_dataset import list_class_folder_samples
from src.datasets.dataset_registry import get_dataset_descriptor, registered_datasets
from src.datasets.split_generator import generate_split_files
from src.features.feature_cache import load_feature_cache, make_fake_feature_cache, save_feature_cache
from src.logging.experiment_logger import finish_experiment_run, start_experiment_run
from src.utils.io import read_json
from src.utils.seed import set_seed
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local CPU smoke validation.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-mode", default="smoke_test")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def create_fake_dataset(root: Path, classes: int = 3, samples_per_class: int = 20) -> None:
    for class_idx in range(classes):
        class_dir = root / f"class_{class_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(samples_per_class):
            (class_dir / f"sample_{sample_idx:03d}.jpg").write_text("fake image placeholder\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.execution_env != "local_wsl" or args.device != "cpu":
        raise SystemExit("Smoke test must run as local_wsl on cpu.")
    if not args.dry_run:
        raise SystemExit("Smoke test requires --dry-run.")

    config = load_configs(["configs/env/local_wsl.yaml", "configs/methods/zero_shot_clip.yaml"])
    validate_required_fields(config, ["execution_env", "run_mode", "device", "is_paper_result", "method.name"])
    set_seed(args.seed, deterministic=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rs_mp_fake_dataset_") as temp_root:
        dataset_root = Path(temp_root)
        create_fake_dataset(dataset_root)
        assert "eurosat" in registered_datasets()
        descriptor = get_dataset_descriptor("eurosat", root=dataset_root)
        samples, class_to_idx = list_class_folder_samples(descriptor)
        if len(samples) != 60 or len(class_to_idx) != 3:
            raise RuntimeError("Dataset registry or class-folder reading failed.")

        split_stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
        split_dir = output_dir / "splits" / f"smoke_seed{args.seed}_{split_stamp}"
        written_splits = generate_split_files(
            dataset="eurosat",
            root=dataset_root,
            output_dir=split_dir,
            shots=[1, 2, 4, 8],
            seeds=[args.seed],
            source_script="scripts/run_smoke_test.py",
        )
        split_path = split_dir / f"shot_1_seed{args.seed}.json"
        split_payload = read_json(split_path)
        if split_payload["shot"] != 1 or len(split_payload["support"]) != 3:
            raise RuntimeError("Few-shot support split validation failed.")

    cache_path = output_dir / "fake_feature_cache.pt"
    cache = make_fake_feature_cache(seed=args.seed)
    if cache_path.exists():
        cache_path = output_dir / f"fake_feature_cache_seed{args.seed}_{len(list(output_dir.glob('fake_feature_cache*.pt')))}.pt"
    save_feature_cache(cache, cache_path)
    loaded_cache = load_feature_cache(cache_path)
    result = ZeroShotClassifier().evaluate(loaded_cache)

    run, metadata = start_experiment_run(
        output_dir=output_dir,
        config=config,
        config_path="configs/methods/zero_shot_clip.yaml",
        dataset="eurosat",
        backbone="fake_backbone",
        method="zero_shot_clip",
        shot=1,
        seed=args.seed,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        split_path=split_path,
        is_paper_result=False,
    )
    metrics = {
        "method": "zero_shot_clip",
        "backbone": "fake_backbone",
        "dataset": "eurosat",
        "shot": 1,
        "seed": args.seed,
        "device": args.device,
        "top1_acc": result.top1_acc,
        "num_samples": result.num_samples,
        "num_classes": result.num_classes,
        "cache_entries": 0,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "inference_time_sec": result.inference_time_sec,
        "images_per_second": result.images_per_second,
        "gpu_memory_mb": None,
        "uses_fake_data": True,
        "uses_fake_features": True,
        "fake_or_dry_run": True,
        "used_fake_features": True,
        "is_real_evaluation": False,
        "config_path": "configs/methods/zero_shot_clip.yaml",
        "config_snapshot_path": str(run.config_snapshot_path),
        "split_path": str(split_path),
        "checkpoint_path": None,
        "prediction_path": None,
        "log_path": str(run.log_path),
        "written_split_count": len(written_splits),
        "feature_cache_path": str(cache_path),
    }
    metadata_path, metrics_path = finish_experiment_run(run, metadata, metrics)
    metadata_json = read_json(metadata_path)
    metrics_json = read_json(metrics_path)
    if metadata_json["is_paper_result"] or metrics_json["is_paper_result"]:
        raise RuntimeError("Local smoke output was incorrectly marked as paper result.")
    print(f"metadata_path={metadata_path}")
    print(f"metrics_path={metrics_path}")
    print(f"feature_cache_path={cache_path}")
    print(f"split_dir={split_dir}")


if __name__ == "__main__":
    main()
