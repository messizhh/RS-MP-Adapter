#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
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
from src.utils.io import read_json, safe_write_csv
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
        "top1_acc": result.metrics["top1_acc"],
        "num_samples": result.metrics["num_samples"],
        "num_classes": result.metrics["num_classes"],
        "per_class_acc": result.metrics["per_class_acc"],
        "confusion_matrix": result.metrics["confusion_matrix"],
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
        "prediction_path": "",
        "per_class_accuracy_path": "",
        "confusion_matrix_path": "",
        "log_path": str(run.log_path),
        "written_split_count": len(written_splits),
        "feature_cache_path": str(cache_path),
    }
    prediction_path = write_prediction_csv(run.run_dir / "predictions.csv", loaded_cache, result.predictions)
    per_class_path = safe_write_csv(
        run.run_dir / "per_class_accuracy.csv",
        result.metrics["per_class_acc"],
        ["class_name", "class_idx", "num_samples", "num_correct", "accuracy"],
    )
    confusion_path = write_confusion_matrix_csv(run.run_dir / "confusion_matrix.csv", result.metrics["confusion_matrix"])
    metrics["prediction_path"] = str(prediction_path)
    metrics["per_class_accuracy_path"] = str(per_class_path)
    metrics["confusion_matrix_path"] = str(confusion_path)
    metadata_path, metrics_path = finish_experiment_run(run, metadata, metrics)
    metadata_json = read_json(metadata_path)
    metrics_json = read_json(metrics_path)
    if metadata_json["is_paper_result"] or metrics_json["is_paper_result"]:
        raise RuntimeError("Local smoke output was incorrectly marked as paper result.")
    table_dir = output_dir / "tables" / run.run_id
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_tables.py",
            "--input-dir",
            str(output_dir),
            "--output-dir",
            str(table_dir),
            "--tables",
            "main",
            "efficiency",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    table_summary_path = parse_output_path(completed.stdout, "summary_path")
    table_summary = read_json(table_summary_path)
    if table_summary["num_eligible_results"] != 0:
        raise RuntimeError("Smoke results were not excluded by export_tables.py")
    print(f"metadata_path={metadata_path}")
    print(f"metrics_path={metrics_path}")
    print(f"prediction_path={prediction_path}")
    print(f"per_class_accuracy_path={per_class_path}")
    print(f"confusion_matrix_path={confusion_path}")
    print(f"feature_cache_path={cache_path}")
    print(f"split_dir={split_dir}")
    print(f"table_summary_path={table_summary_path}")
    for key, value in table_summary.get("outputs", {}).items():
        print(f"table_{key}_path={value}")


def write_prediction_csv(path: Path, cache, predictions: list[int]) -> Path:
    rows = []
    for index, (image_path, label, prediction) in enumerate(zip(cache.image_paths, cache.image_labels, predictions)):
        rows.append(
            {
                "sample_id": index,
                "path": image_path,
                "label": int(label),
                "pred": int(prediction),
                "correct": int(label == prediction),
                "split": cache.split_name,
            }
        )
    return safe_write_csv(path, rows, ["sample_id", "path", "label", "pred", "correct", "split"])


def write_confusion_matrix_csv(path: Path, matrix: list[list[int]]) -> Path:
    rows = []
    for label_idx, row in enumerate(matrix):
        for pred_idx, count in enumerate(row):
            rows.append({"label": label_idx, "pred": pred_idx, "count": count})
    return safe_write_csv(path, rows, ["label", "pred", "count"])


def parse_output_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise RuntimeError(f"Could not find {key} in command output: {stdout}")


if __name__ == "__main__":
    main()
