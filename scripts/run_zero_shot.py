#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_zero_shot_eval_preflight import (
    make_base_request,
    resolve_cache_path,
    resolve_manifest_entries,
    select_base_entries,
)
from src.baselines.zero_shot import ZeroShotClassifier
from src.config.config_loader import load_configs, save_config_snapshot
from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import FeatureCache, load_feature_cache, make_fake_feature_cache, shape_of_2d, to_labels
from src.logging.experiment_logger import create_run_id, is_paper_allowed
from src.logging.result_schema import validate_metadata_schema, validate_metrics_schema
from src.logging.system_info import get_system_info
from src.utils.features import argmax_rows
from src.utils.io import read_json, safe_write_csv, safe_write_json, write_json_no_overwrite
from src.utils.seed import set_seed
from src.utils.timing import utc_now_iso


SERVER_RUN_MODES = {"server_full", "server_ablation", "server_benchmark"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached zero-shot evaluation over image and text feature caches.")
    parser.add_argument("--config", default="configs/methods/zero_shot_clip.yaml")
    parser.add_argument("--env-config", default="configs/env/local_wsl.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default="zero_shot")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="", help="Legacy CLI option; use --base-split for cached mode.")
    parser.add_argument("--base-split", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--feature-cache", default="", help="Legacy single-cache option; cached mode uses --manifest.")
    parser.add_argument("--text-feature-cache", default="")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution-env", default=None)
    parser.add_argument("--run-mode", default=None)
    parser.add_argument("--preflight-report", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--allow-paper-result", action="store_true")
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [path for path in [args.env_config, args.config] if path]
    config = load_configs(config_paths, args.override)
    execution_env = args.execution_env or config.get("execution_env", "local_wsl")
    run_mode = args.run_mode or config.get("run_mode", "local_validation")
    device = args.device or config.get("device", "cpu")
    result = run_cached_zero_shot_evaluation(
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        backbone=args.backbone,
        base_split=args.base_split or args.split,
        manifest_path=args.manifest,
        text_feature_cache_path=args.text_feature_cache,
        eval_splits=args.eval_splits,
        output_dir=args.output_dir,
        device=device,
        execution_env=execution_env,
        run_mode=run_mode,
        preflight_report=args.preflight_report,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        seed=args.seed,
        temperature=args.temperature,
        normalize_features=not args.no_normalize_features,
        save_predictions=args.save_predictions,
        allow_paper_result=args.allow_paper_result,
        skip_preflight_check=args.skip_preflight_check,
        command=shlex.join(sys.argv),
    )
    print(f"run_dir={result['run_dir']}")
    print(f"metadata_path={result['metadata_path']}")
    print(f"metrics_path={result['metrics_path']}")
    if result.get("prediction_path"):
        print(f"prediction_path={result['prediction_path']}")


def run_cached_zero_shot_evaluation(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    dataset: str,
    backbone: str,
    base_split: str,
    manifest_path: str | Path | None,
    text_feature_cache_path: str | Path | None,
    eval_splits: list[str],
    output_dir: str | Path,
    device: str,
    execution_env: str,
    run_mode: str,
    preflight_report: str | Path | None,
    dry_run: bool,
    max_samples: int | None,
    seed: int,
    temperature: float = 1.0,
    normalize_features: bool = True,
    save_predictions: bool = False,
    allow_paper_result: bool = False,
    skip_preflight_check: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    set_seed(seed, deterministic=True)
    output_root = Path(output_dir)
    ensure_results_output_allowed(output_root)
    eval_splits = list(dict.fromkeys(eval_splits))
    if not eval_splits:
        raise ValueError("--eval-splits must contain at least one split")
    if not dry_run:
        validate_preflight_report(preflight_report, skip_preflight_check=skip_preflight_check)

    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    if dry_run and not manifest_path:
        image_caches = make_dry_run_image_caches(dataset=dataset, backbone=backbone, seed=seed, max_samples=max_samples, eval_splits=eval_splits)
        text_cache = {
            "text_features": image_caches[eval_splits[0]].text_features,
            "class_names": image_caches[eval_splits[0]].class_names,
            "class_to_idx": image_caches[eval_splits[0]].class_to_idx,
            "feature_dim": image_caches[eval_splits[0]].feature_dim,
            "num_classes": len(image_caches[eval_splits[0]].class_to_idx),
            "dataset": dataset,
            "backbone": backbone,
            "base_split": base_split or "dry_run",
            "dry_run": True,
            "uses_fake_text_features": True,
            "is_paper_result": False,
        }
        image_cache_paths = {split: "" for split in eval_splits}
        effective_base_split = base_split or "dry_run"
    else:
        if not manifest_path:
            raise ValueError("cached zero-shot evaluation requires --manifest unless --dry-run is used without real caches")
        if not text_feature_cache_path:
            raise ValueError("cached zero-shot evaluation requires --text-feature-cache")
        if not base_split:
            raise ValueError("cached zero-shot evaluation requires --base-split")
        image_caches, image_cache_paths, effective_base_split = load_image_caches_from_manifest(
            manifest_path=Path(manifest_path),
            dataset=dataset,
            backbone=backbone,
            base_split=base_split,
            eval_splits=eval_splits,
        )
        text_cache = load_text_feature_cache(Path(text_feature_cache_path))

    reference_cache = image_caches[eval_splits[0]]
    text_features, class_names, feature_dim, num_classes = validate_text_cache_for_evaluation(
        text_cache=text_cache,
        image_cache=reference_cache,
        dataset=dataset,
        backbone=backbone,
        base_split=effective_base_split,
        dry_run=dry_run,
    )
    classifier = ZeroShotClassifier(temperature=temperature, normalize_features=normalize_features)
    classifier.fit(text_features)
    classifier.class_names = class_names

    per_split_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    total_samples = 0
    inference_time_sec = 0.0
    for split in eval_splits:
        cache = image_caches[split]
        validate_image_cache_for_evaluation(cache, dataset=dataset, backbone=backbone, class_names=class_names, feature_dim=feature_dim)
        split_start = time.perf_counter()
        logits = classifier.predict_logits(cache.image_features)
        split_elapsed = time.perf_counter() - split_start
        labels = to_labels(cache.image_labels)
        metrics_core = evaluate_logits(logits, labels, class_names=class_names)
        predictions = argmax_rows(logits)
        per_split_metrics[split] = {
            "top1_acc": metrics_core["top1_acc"],
            "num_samples": metrics_core["num_samples"],
            "num_classes": metrics_core["num_classes"],
            "per_class_acc": metrics_core["per_class_acc"],
            "confusion_matrix": metrics_core["confusion_matrix"],
            "inference_time_sec": split_elapsed,
            "images_per_second": float(len(labels) / split_elapsed) if split_elapsed > 0 else 0.0,
        }
        total_samples += len(labels)
        inference_time_sec += split_elapsed
        if save_predictions:
            prediction_rows.extend(prediction_csv_rows(split, cache.image_paths, labels, predictions))

    is_paper_result = is_paper_allowed(execution_env, run_mode, allow_paper_result and run_mode in SERVER_RUN_MODES)
    eligible_for_paper_tables = bool(is_paper_result and run_mode in SERVER_RUN_MODES)
    run_id, run_dir = create_zero_shot_run_dir(output_root, dataset=dataset, backbone=backbone, seed=seed)
    run_config = {
        **config,
        "zero_shot_cached_evaluation": {
            "dataset": dataset,
            "backbone": backbone,
            "base_split": effective_base_split,
            "eval_splits": eval_splits,
            "manifest_path": str(manifest_path or ""),
            "text_feature_cache_path": str(text_feature_cache_path or ""),
            "dry_run": dry_run,
            "save_predictions": save_predictions,
        },
    }
    config_snapshot_path = save_config_snapshot(run_config, run_dir)
    log_path = run_dir / "log.txt"
    log_path.write_text("Cached zero-shot evaluation initialized.\n", encoding="utf-8")
    prediction_path = ""
    if save_predictions:
        prediction_path = str(safe_write_csv(run_dir / "predictions.csv", prediction_rows, ["sample_id", "split", "path", "label", "pred", "correct"]))

    end_time = utc_now_iso()
    system_info = get_system_info(device=device)
    top1_by_split = {split: payload["top1_acc"] for split, payload in per_split_metrics.items()}
    primary_split = "test" if "test" in per_split_metrics else eval_splits[-1]
    fake_text = bool(text_cache.get("dry_run", False) or text_cache.get("uses_fake_text_features", False))
    fake_images = any(bool(cache.metadata.get("uses_fake_features", False) or cache.metadata.get("uses_fake_data", False)) for cache in image_caches.values())
    metadata = {
        **system_info,
        "run_id": run_id,
        "command": command or shlex.join(sys.argv),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "seed": seed,
        "dataset": dataset,
        "shot": None,
        "backbone": backbone,
        "method": "zero_shot",
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": device,
        "server_job_id": None,
        "base_split": effective_base_split,
        "eval_splits": eval_splits,
        "split_path": effective_base_split,
        "image_cache_paths": image_cache_paths,
        "text_feature_cache_path": str(text_feature_cache_path or ""),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "start_time": start_time,
        "end_time": end_time,
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": save_predictions,
        "writes_results_raw": True,
    }
    metrics = {
        "run_id": run_id,
        "method": "zero_shot",
        "backbone": backbone,
        "dataset": dataset,
        "shot": None,
        "seed": seed,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": device,
        "base_split": effective_base_split,
        "eval_splits": eval_splits,
        "image_cache_paths": image_cache_paths,
        "text_feature_cache_path": str(text_feature_cache_path or ""),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "top1_acc": top1_by_split.get(primary_split),
        "top1_acc_by_split": top1_by_split,
        "per_split": per_split_metrics,
        "num_samples": total_samples,
        "cache_entries": num_classes,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "inference_time_sec": inference_time_sec,
        "images_per_second": float(total_samples / inference_time_sec) if inference_time_sec > 0 else 0.0,
        "gpu_memory_mb": None,
        "uses_fake_data": bool(dry_run or fake_images),
        "uses_fake_features": bool(dry_run or fake_images or fake_text),
        "fake_or_dry_run": bool(dry_run or fake_images or fake_text),
        "used_fake_features": bool(dry_run or fake_images or fake_text),
        "is_real_evaluation": not bool(dry_run or fake_images or fake_text),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "split_path": effective_base_split,
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
        "prediction_path": prediction_path,
        "checkpoint_path": None,
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": save_predictions,
        "writes_results_raw": True,
        "start_time": start_time,
        "end_time": end_time,
        "total_time_sec": time.perf_counter() - start_perf,
    }
    validate_metadata_schema(metadata)
    validate_metrics_schema(metrics)
    metadata_path = write_json_no_overwrite(run_dir / "metadata.json", metadata)
    metrics_path = write_json_no_overwrite(run_dir / "metrics.json", metrics)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"Cached zero-shot evaluation finished at {end_time}.\n")
    return {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "prediction_path": Path(prediction_path) if prediction_path else None,
        "metrics": metrics,
    }


def validate_preflight_report(preflight_report: str | Path | None, *, skip_preflight_check: bool) -> None:
    if skip_preflight_check:
        return
    if not preflight_report:
        raise ValueError("cached zero-shot evaluation requires --preflight-report or --skip-preflight-check")
    report = read_json(preflight_report)
    if report.get("is_valid") is not True or report.get("zero_shot_input_ready") is not True:
        raise ValueError("zero-shot preflight report is not ready; pass --skip-preflight-check only for controlled debugging")


def load_image_caches_from_manifest(
    *,
    manifest_path: Path,
    dataset: str,
    backbone: str,
    base_split: str,
    eval_splits: list[str],
) -> tuple[dict[str, FeatureCache], dict[str, str], str]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_json(manifest_path)
    entries = resolve_manifest_entries(manifest, manifest_path, errors)
    if errors:
        raise ValueError("; ".join(errors))
    selected_entries = [entry for entry in entries if entry.get("dataset") == dataset and entry.get("backbone") == backbone]
    base_request = make_base_request(base_split, dataset)
    base_entries = select_base_entries(selected_entries, base_request, warnings)
    caches: dict[str, FeatureCache] = {}
    cache_paths: dict[str, str] = {}
    for split in eval_splits:
        entry = base_entries.get(split)
        if entry is None:
            raise ValueError(f"{base_split} is missing required {split} feature cache")
        cache_path = resolve_cache_path(entry, Path(str(entry.get("summary_path", "."))).parent)
        if cache_path is None or not cache_path.exists():
            raise ValueError(f"{split} feature cache does not exist: {cache_path}")
        cache = load_feature_cache(cache_path)
        caches[split] = cache
        cache_paths[split] = str(cache_path)
    return caches, cache_paths, base_request["split_id"]


def make_dry_run_image_caches(
    *, dataset: str, backbone: str, seed: int, max_samples: int | None, eval_splits: list[str]
) -> dict[str, FeatureCache]:
    caches = {}
    for index, split in enumerate(eval_splits):
        caches[split] = make_fake_feature_cache(
            num_samples=max_samples or 12,
            num_classes=3,
            feature_dim=8,
            seed=seed + index,
            split_name=split,
            dataset=dataset,
            backbone=backbone,
        )
    return caches


def load_text_feature_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"text feature cache does not exist: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("text feature cache must contain a mapping")
    return data


def validate_text_cache_for_evaluation(
    *,
    text_cache: dict[str, Any],
    image_cache: FeatureCache,
    dataset: str,
    backbone: str,
    base_split: str,
    dry_run: bool,
) -> tuple[Any, list[str], int, int]:
    if text_cache.get("dataset") != dataset:
        raise ValueError(f"text cache dataset mismatch, expected {dataset}, found {text_cache.get('dataset')}")
    if text_cache.get("backbone") != backbone:
        raise ValueError(f"text cache backbone mismatch, expected {backbone}, found {text_cache.get('backbone')}")
    if str(text_cache.get("base_split", "")) not in {base_split, Path(base_split).stem, "dry_run"}:
        raise ValueError(f"text cache base_split mismatch, expected {base_split}, found {text_cache.get('base_split')}")
    if text_cache.get("is_paper_result") is not False:
        raise ValueError("text cache must have is_paper_result=false")
    if not dry_run and (bool(text_cache.get("dry_run", False)) or bool(text_cache.get("uses_fake_text_features", False))):
        raise ValueError("dry-run/fake text cache is not allowed for non-dry-run cached zero-shot evaluation")
    text_features = text_cache.get("text_features")
    shape = list(shape_of_2d(text_features))
    class_names = [str(item) for item in text_cache.get("class_names", [])] if isinstance(text_cache.get("class_names"), list) else []
    class_to_idx = text_cache.get("class_to_idx")
    if not class_names:
        raise ValueError("text cache must contain class_names")
    expected_class_names = image_cache.class_names
    if class_names != expected_class_names:
        raise ValueError("text cache class_names do not match image cache class order")
    if class_to_idx != image_cache.class_to_idx:
        raise ValueError("text cache class_to_idx does not match image cache class_to_idx")
    feature_dim = int(text_cache.get("feature_dim") or image_cache.feature_dim)
    num_classes = int(text_cache.get("num_classes") or len(class_names))
    if shape != [num_classes, feature_dim]:
        raise ValueError(f"text_features shape {shape} does not equal expected [{num_classes}, {feature_dim}]")
    return text_features, class_names, feature_dim, num_classes


def validate_image_cache_for_evaluation(
    cache: FeatureCache, *, dataset: str, backbone: str, class_names: list[str], feature_dim: int
) -> None:
    cache.validate()
    if cache.dataset != dataset:
        raise ValueError(f"image cache dataset mismatch, expected {dataset}, found {cache.dataset}")
    if cache.backbone != backbone:
        raise ValueError(f"image cache backbone mismatch, expected {backbone}, found {cache.backbone}")
    if cache.class_names != class_names:
        raise ValueError("image cache class order does not match text cache class order")
    if cache.feature_dim != feature_dim:
        raise ValueError(f"image cache feature_dim={cache.feature_dim} does not match text feature_dim={feature_dim}")


def create_zero_shot_run_dir(output_dir: Path, *, dataset: str, backbone: str, seed: int) -> tuple[str, Path]:
    base = output_dir / dataset / backbone / "zero_shot" / f"seed_{seed}"
    for _ in range(100):
        run_id = create_run_id()
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique zero-shot run directory under {base}")


def prediction_csv_rows(split: str, image_paths: list[str], labels: list[int], predictions: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index, (image_path, label, prediction) in enumerate(zip(image_paths, labels, predictions)):
        rows.append(
            {
                "sample_id": index,
                "split": split,
                "path": image_path,
                "label": label,
                "pred": prediction,
                "correct": int(label == prediction),
            }
        )
    return rows


def ensure_results_output_allowed(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "outputs" and parts[index + 1] == "preflight":
            raise ValueError("cached zero-shot evaluation metrics must not be written under outputs/preflight")


if __name__ == "__main__":
    main()
