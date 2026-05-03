from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import FeatureCache, load_feature_cache, make_fake_feature_cache
from src.logging.experiment_logger import finish_experiment_run, start_experiment_run
from src.utils.features import argmax_rows, to_labels, to_rows
from src.utils.io import safe_write_csv


def load_or_make_cache(args) -> FeatureCache:
    if getattr(args, "dry_run", False):
        return make_fake_feature_cache(num_samples=args.max_samples or 12, dataset=args.dataset, backbone=args.backbone)
    if getattr(args, "feature_cache", ""):
        return load_feature_cache(args.feature_cache)
    raise SystemExit("Provide --feature-cache or use --dry-run for local fake validation.")


def split_support_query(cache: FeatureCache, shot: int = 1):
    rows = to_rows(cache.image_features)
    labels = to_labels(cache.image_labels)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    support_idx = []
    for label in sorted(by_class):
        if len(by_class[label]) < shot:
            raise ValueError(f"Class {label} has fewer samples than shot={shot}")
        support_idx.extend(by_class[label][:shot])
    support_set = set(support_idx)
    query_idx = [index for index in range(len(rows)) if index not in support_set]
    if not query_idx:
        query_idx = support_idx
    return {
        "support_features": [rows[index] for index in support_idx],
        "support_labels": [labels[index] for index in support_idx],
        "query_features": [rows[index] for index in query_idx],
        "query_labels": [labels[index] for index in query_idx],
        "query_paths": [cache.image_paths[index] for index in query_idx],
    }


def write_prediction_csv(path: Path, image_paths: list[str], labels: list[int], predictions: list[int], split_name: str = "query") -> Path:
    rows = []
    for index, (image_path, label, prediction) in enumerate(zip(image_paths, labels, predictions)):
        rows.append({"sample_id": index, "path": image_path, "label": label, "pred": prediction, "correct": int(label == prediction), "split": split_name})
    return safe_write_csv(path, rows, ["sample_id", "path", "label", "pred", "correct", "split"])


def run_training_free_method(args, config: dict[str, Any], method, cache: FeatureCache, extra_metrics: dict[str, Any] | None = None):
    split = split_support_query(cache, shot=args.shot)
    method.class_names = cache.class_names
    if getattr(method, "text_features", None) is None and getattr(args, "use_text_fusion", False):
        method.text_features = cache.text_features
    start = time.perf_counter()
    method.fit(split["support_features"], split["support_labels"])
    logits = method.predict_logits(split["query_features"])
    elapsed = time.perf_counter() - start
    metrics_core = evaluate_logits(logits, split["query_labels"], class_names=cache.class_names)
    predictions = argmax_rows(logits)
    run, metadata = start_experiment_run(
        output_dir=args.output_dir,
        config=config,
        config_path=getattr(args, "config", None) or "",
        dataset=args.dataset,
        backbone=args.backbone,
        method=args.method,
        shot=args.shot,
        seed=args.seed,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        device=args.device,
        split_path=getattr(args, "split", ""),
        is_paper_result=False,
    )
    prediction_path = write_prediction_csv(run.run_dir / "predictions.csv", split["query_paths"], split["query_labels"], predictions)
    metrics = {
        "method": args.method,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "device": args.device,
        "top1_acc": metrics_core["top1_acc"],
        "num_samples": metrics_core["num_samples"],
        "num_classes": metrics_core["num_classes"],
        "per_class_acc": metrics_core["per_class_acc"],
        "confusion_matrix": metrics_core["confusion_matrix"],
        "cache_entries": int(getattr(method, "cache_entries", 0)),
        "trainable_params": int(getattr(method, "trainable_params", 0)),
        "training_time_sec": 0.0,
        "inference_time_sec": elapsed,
        "images_per_second": float(len(split["query_labels"]) / elapsed) if elapsed > 0 else 0.0,
        "gpu_memory_mb": None,
        "uses_fake_data": bool(cache.metadata.get("uses_fake_data", args.dry_run)),
        "uses_fake_features": bool(cache.metadata.get("uses_fake_features", args.dry_run)),
        "fake_or_dry_run": bool(args.dry_run or cache.metadata.get("uses_fake_features", False)),
        "is_real_evaluation": not bool(args.dry_run or cache.metadata.get("uses_fake_features", False)),
        "feature_cache_path": getattr(args, "feature_cache", ""),
        "prediction_path": str(prediction_path),
        "checkpoint_path": None,
        "log_path": str(run.log_path),
    }
    if hasattr(method, "compression_ratio"):
        metrics["compression_ratio"] = float(method.compression_ratio)
    if hasattr(method, "compression_info"):
        metrics.update(
            {
                "original_cache_entries": method.compression_info.get("original_cache_entries", 0),
                "compressed_cache_entries": method.compression_info.get("compressed_cache_entries", 0),
            }
        )
    if extra_metrics:
        metrics.update(extra_metrics)
    metadata_path, metrics_path = finish_experiment_run(run, metadata, metrics)
    return {"metadata_path": metadata_path, "metrics_path": metrics_path, "prediction_path": prediction_path, "metrics": metrics}
