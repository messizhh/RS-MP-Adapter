#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cached_adapter_runner import (
    SERVER_RUN_MODES,
    find_split_summary,
    load_support_cache_from_manifest,
    make_dry_run_adapter_caches,
    make_dry_run_text_cache,
    split_tokens,
    validate_support_cache_for_adapter,
)
from scripts.run_zero_shot import (
    load_image_caches_from_manifest,
    load_text_feature_cache,
    prediction_csv_rows,
    validate_image_cache_for_evaluation,
    validate_text_cache_for_evaluation,
)
from src.adapters.rs_cpc_adapter import RsCpcAdapter
from src.config.config_loader import load_configs, save_config_snapshot
from src.eval.evaluator import evaluate_logits
from src.features.feature_cache import to_labels
from src.logging.experiment_logger import create_run_id, is_paper_allowed
from src.logging.result_schema import validate_metadata_schema, validate_metrics_schema
from src.logging.system_info import get_system_info
from src.utils.features import argmax_rows
from src.utils.io import read_json, safe_write_csv, write_json_no_overwrite
from src.utils.seed import set_seed
from src.utils.timing import utc_now_iso


SUPPORTED_PROTOTYPE_INITS = {"mean", "random_group_mean", "medoid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached training-free RS-CPC evaluation.")
    parser.add_argument("--config", default="configs/methods/rs_cpc.yaml")
    parser.add_argument("--env-config", default="configs/env/local_wsl.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--method", default="rs_cpc")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--M", "--num-prototypes-per-class", dest="m_value", type=int, default=1)
    parser.add_argument("--prototype-init", default="random_group_mean", choices=["mean", "random_group_mean", "medoid", "kmeans"])
    parser.add_argument("--split", default="", help="Legacy CLI option; use --shot-split for cached mode.")
    parser.add_argument("--feature-cache", default="", help="Legacy CLI option; cached mode uses --manifest.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--base-split", default="")
    parser.add_argument("--shot-split", default="")
    parser.add_argument("--text-feature-cache", default="")
    parser.add_argument("--adapter-input-plan", default="")
    parser.add_argument("--prototype-preflight-report", default="")
    parser.add_argument("--preflight-report", default="")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution-env", default=None)
    parser.add_argument("--run-mode", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--allow-paper-result", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fusion", default="fixed_alpha")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finetune:
        raise SystemExit("Fine-tuned RS-CPC is disabled for this cached training-free runner.")
    config_paths = [path for path in [args.env_config, args.config] if path]
    config = load_configs(config_paths, args.override)
    result = run_cached_rs_cpc_evaluation(
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        backbone=args.backbone,
        shot=args.shot,
        seed=args.seed,
        m_value=args.m_value,
        prototype_init=args.prototype_init,
        manifest_path=args.manifest,
        base_split=args.base_split,
        shot_split=args.shot_split or args.split,
        text_feature_cache_path=args.text_feature_cache,
        adapter_input_plan=args.adapter_input_plan,
        prototype_preflight_report=args.prototype_preflight_report,
        preflight_report=args.preflight_report,
        eval_splits=args.eval_splits,
        output_dir=args.output_dir,
        device=args.device or config.get("device", "cpu"),
        execution_env=args.execution_env or config.get("execution_env", "local_wsl"),
        run_mode=args.run_mode or config.get("run_mode", "local_validation"),
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        alpha=args.alpha,
        temperature=args.temperature,
        fusion=args.fusion,
        save_predictions=args.save_predictions,
        allow_paper_result=args.allow_paper_result,
        command=shlex.join(sys.argv),
    )
    print(f"run_dir={result['run_dir']}")
    print(f"metadata_path={result['metadata_path']}")
    print(f"metrics_path={result['metrics_path']}")
    if result.get("prediction_path"):
        print(f"prediction_path={result['prediction_path']}")


def run_cached_rs_cpc_evaluation(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    dataset: str,
    backbone: str,
    shot: int,
    seed: int,
    m_value: int,
    prototype_init: str,
    manifest_path: str | Path | None,
    base_split: str,
    shot_split: str,
    text_feature_cache_path: str | Path | None,
    adapter_input_plan: str | Path | None,
    prototype_preflight_report: str | Path | None,
    preflight_report: str | Path | None,
    eval_splits: list[str],
    output_dir: str | Path,
    device: str,
    execution_env: str,
    run_mode: str,
    dry_run: bool,
    max_samples: int | None,
    alpha: float = 1.0,
    temperature: float = 1.0,
    fusion: str = "fixed_alpha",
    save_predictions: bool = False,
    allow_paper_result: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    validate_rs_cpc_request(shot=shot, m_value=m_value, prototype_init=prototype_init)
    if execution_env == "local_wsl" and device != "cpu":
        raise ValueError("Local WSL cached RS-CPC runs must use --device cpu.")
    set_seed(seed, deterministic=True)
    output_root = Path(output_dir)
    ensure_results_output_allowed(output_root)
    eval_splits = list(dict.fromkeys(eval_splits))
    if not eval_splits:
        raise ValueError("--eval-splits must contain at least one split")

    uses_real_cache_inputs = bool(
        manifest_path or text_feature_cache_path or adapter_input_plan or preflight_report or prototype_preflight_report
    )
    if uses_real_cache_inputs or not dry_run:
        if not manifest_path:
            raise ValueError("cached RS-CPC evaluation requires --manifest")
        if not base_split:
            raise ValueError("cached RS-CPC evaluation requires --base-split")
        if not shot_split:
            raise ValueError("cached RS-CPC evaluation requires --shot-split")
        if not text_feature_cache_path:
            raise ValueError("cached RS-CPC evaluation requires --text-feature-cache")
        validate_rs_cpc_adapter_preflight(
            preflight_report=preflight_report,
            dataset=dataset,
            backbone=backbone,
            base_split=base_split,
            shot_split=shot_split,
            shot=shot,
            m_value=m_value,
            eval_splits=eval_splits,
        )
        validate_rs_cpc_adapter_input_plan(
            adapter_input_plan=adapter_input_plan,
            dataset=dataset,
            backbone=backbone,
            shot=shot,
            m_value=m_value,
            shot_split=shot_split,
        )
        validate_rs_cpc_prototype_preflight(
            prototype_preflight_report=prototype_preflight_report,
            dataset=dataset,
            backbone=backbone,
            shot=shot,
            m_value=m_value,
            shot_split=shot_split,
            prototype_init=prototype_init,
        )

    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    if dry_run and not manifest_path:
        support_cache, image_caches, image_cache_paths, support_cache_path, effective_base_split, effective_shot_split = (
            make_dry_run_adapter_caches(
                dataset=dataset,
                backbone=backbone,
                seed=seed,
                shot=shot,
                max_samples=max_samples,
                eval_splits=eval_splits,
            )
        )
        text_cache = make_dry_run_text_cache(reference_cache=support_cache, dataset=dataset, backbone=backbone)
    else:
        support_cache, support_cache_path, effective_shot_split = load_support_cache_from_manifest(
            manifest_path=Path(str(manifest_path)),
            dataset=dataset,
            backbone=backbone,
            shot_split=shot_split,
        )
        image_caches, image_cache_paths, effective_base_split = load_image_caches_from_manifest(
            manifest_path=Path(str(manifest_path)),
            dataset=dataset,
            backbone=backbone,
            base_split=base_split,
            eval_splits=eval_splits,
        )
        text_cache = load_text_feature_cache(Path(str(text_feature_cache_path)))

    reference_cache = image_caches[eval_splits[0]]
    text_features, class_names, feature_dim, num_classes = validate_text_cache_for_evaluation(
        text_cache=text_cache,
        image_cache=reference_cache,
        dataset=dataset,
        backbone=backbone,
        base_split=effective_base_split,
        dry_run=dry_run,
    )
    validate_image_cache_for_evaluation(
        support_cache,
        dataset=dataset,
        backbone=backbone,
        class_names=class_names,
        feature_dim=feature_dim,
    )
    validate_support_cache_for_adapter(support_cache, shot=shot, num_classes=num_classes)

    method = RsCpcAdapter(
        num_prototypes_per_class=m_value,
        prototype_init=prototype_init,
        seed=seed,
        temperature=temperature,
        alpha=alpha,
        text_features=text_features,
    )
    method.class_names = class_names
    method.fit(support_cache.image_features, support_cache.image_labels)

    per_split_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    total_samples = 0
    inference_time_sec = 0.0
    for split in eval_splits:
        cache = image_caches[split]
        validate_image_cache_for_evaluation(cache, dataset=dataset, backbone=backbone, class_names=class_names, feature_dim=feature_dim)
        split_start = time.perf_counter()
        logits = method.predict_logits(cache.image_features)
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

    expected_cache_entries = num_classes * m_value
    if method.cache_entries != expected_cache_entries:
        raise ValueError(f"RS-CPC cache_entries={method.cache_entries} does not equal num_classes*M={expected_cache_entries}")

    is_paper_result = is_paper_allowed(execution_env, run_mode, allow_paper_result and run_mode in SERVER_RUN_MODES)
    eligible_for_paper_tables = bool(is_paper_result and run_mode in SERVER_RUN_MODES)
    run_id, run_dir = create_rs_cpc_run_dir(
        output_root,
        dataset=dataset,
        backbone=backbone,
        shot=shot,
        m_value=m_value,
        prototype_init=prototype_init,
        seed=seed,
    )
    run_config = {
        **config,
        "cached_rs_cpc_evaluation": {
            "method": "rs_cpc",
            "dataset": dataset,
            "backbone": backbone,
            "shot": shot,
            "seed": seed,
            "M": m_value,
            "prototype_init": prototype_init,
            "base_split": effective_base_split,
            "shot_split": effective_shot_split,
            "eval_splits": eval_splits,
            "manifest_path": str(manifest_path or ""),
            "support_cache_path": support_cache_path,
            "image_cache_paths": image_cache_paths,
            "text_feature_cache_path": str(text_feature_cache_path or ""),
            "adapter_input_plan": str(adapter_input_plan or ""),
            "prototype_preflight_report": str(prototype_preflight_report or ""),
            "dry_run": dry_run,
            "save_predictions": save_predictions,
            "alpha": alpha,
            "temperature": temperature,
            "fusion": fusion,
        },
    }
    config_snapshot_path = save_config_snapshot(run_config, run_dir)
    log_path = run_dir / "log.txt"
    log_path.write_text("Cached training-free RS-CPC evaluation initialized.\n", encoding="utf-8")
    prediction_path = ""
    if save_predictions:
        prediction_path = str(safe_write_csv(run_dir / "predictions.csv", prediction_rows, ["sample_id", "split", "path", "label", "pred", "correct"]))

    end_time = utc_now_iso()
    system_info = get_system_info(device=device)
    command_text = command or shlex.join(sys.argv)
    top1_by_split = {split: payload["top1_acc"] for split, payload in per_split_metrics.items()}
    primary_split = "test" if "test" in per_split_metrics else eval_splits[-1]
    fake_text = bool(text_cache.get("dry_run", False) or text_cache.get("uses_fake_text_features", False))
    fake_images = bool(
        support_cache.metadata.get("uses_fake_features", False)
        or support_cache.metadata.get("uses_fake_data", False)
        or any(cache.metadata.get("uses_fake_features", False) or cache.metadata.get("uses_fake_data", False) for cache in image_caches.values())
    )
    common_payload = {
        "run_id": run_id,
        "method": "rs_cpc",
        "backbone": backbone,
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "M": m_value,
        "prototype_init": prototype_init,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "device": device,
        "base_split": effective_base_split,
        "shot_split": effective_shot_split,
        "eval_splits": eval_splits,
        "split_path": effective_shot_split,
        "support_cache_path": support_cache_path,
        "image_cache_paths": image_cache_paths,
        "text_feature_cache_path": str(text_feature_cache_path or ""),
        "adapter_input_plan": str(adapter_input_plan or ""),
        "prototype_preflight_report": str(prototype_preflight_report or ""),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "cache_entries": expected_cache_entries,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "top1_acc_by_split": top1_by_split,
        "computes_logits": True,
        "computes_accuracy": True,
        "evaluates_model": True,
        "trains_model": False,
        "extracts_features": False,
        "loads_model": False,
        "saves_predictions": save_predictions,
        "writes_results_raw": True,
    }
    metadata = {
        **system_info,
        **common_payload,
        "torch_version": system_info.get("pytorch_version"),
        "command": command_text,
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "server_job_id": None,
        "start_time": start_time,
        "end_time": end_time,
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
    }
    metrics = {
        **system_info,
        **common_payload,
        "torch_version": system_info.get("pytorch_version"),
        "command": command_text,
        "top1_acc": top1_by_split.get(primary_split),
        "per_split": per_split_metrics,
        "num_samples": total_samples,
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
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
        "prediction_path": prediction_path,
        "checkpoint_path": None,
        "start_time": start_time,
        "end_time": end_time,
        "total_time_sec": time.perf_counter() - start_perf,
        "alpha": alpha,
        "temperature": temperature,
        "fusion": fusion,
        "original_cache_entries": method.compression_info.get("original_cache_entries", 0),
        "compressed_cache_entries": method.compression_info.get("compressed_cache_entries", 0),
        "compression_ratio": method.compression_ratio,
    }
    validate_metadata_schema(metadata)
    validate_metrics_schema(metrics)
    metadata_path = write_json_no_overwrite(run_dir / "metadata.json", metadata)
    metrics_path = write_json_no_overwrite(run_dir / "metrics.json", metrics)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"Cached training-free RS-CPC evaluation finished at {end_time}.\n")
    return {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "prediction_path": Path(prediction_path) if prediction_path else None,
        "metrics": metrics,
    }


def validate_rs_cpc_request(*, shot: int, m_value: int, prototype_init: str) -> None:
    if shot <= 0:
        raise ValueError("shot must be a positive integer")
    if m_value <= 0:
        raise ValueError("M must be a positive integer")
    if m_value > shot:
        raise ValueError("M must not exceed shot")
    if prototype_init == "mean" and m_value != 1:
        raise ValueError("mean prototype_init supports only M=1")
    if prototype_init == "kmeans":
        raise ValueError("kmeans prototype_init is unsupported in the cached training-free RS-CPC runner")
    if prototype_init not in SUPPORTED_PROTOTYPE_INITS:
        raise ValueError(f"unsupported prototype_init: {prototype_init}")


def validate_rs_cpc_adapter_preflight(
    *,
    preflight_report: str | Path | None,
    dataset: str,
    backbone: str,
    base_split: str,
    shot_split: str,
    shot: int,
    m_value: int,
    eval_splits: list[str],
) -> None:
    if not preflight_report:
        raise ValueError("cached RS-CPC evaluation requires --preflight-report")
    report = read_json(preflight_report)
    if report.get("is_valid") is not True:
        raise ValueError("adapter input preflight report is not ready")
    if report.get("dataset") != dataset:
        raise ValueError(f"adapter input preflight dataset mismatch, expected {dataset}, found {report.get('dataset')}")
    if report.get("backbone") != backbone:
        raise ValueError(f"adapter input preflight backbone mismatch, expected {backbone}, found {report.get('backbone')}")
    base_summary = find_split_summary(report.get("per_split_summary"), base_split)
    if not base_summary:
        raise ValueError("adapter input preflight report is not ready for requested base split")
    for split in eval_splits:
        ready_key = f"{split}_ready_for_evaluation_input" if split == "test" else f"{split}_ready_for_tuning_input"
        section_ready = bool(base_summary.get("sections", {}).get(split, {}).get("is_ready"))
        if not bool(base_summary.get(ready_key, section_ready)):
            raise ValueError(f"adapter input preflight report is not ready for {split} cache")
    shot_summary = find_split_summary(report.get("per_split_summary"), shot_split)
    if not shot_summary:
        raise ValueError("adapter input preflight report is not ready for requested shot split")
    support = shot_summary.get("support") if isinstance(shot_summary, dict) else None
    if not isinstance(support, dict) or support.get("is_ready") is not True:
        raise ValueError("adapter input preflight report is not ready for requested support cache")
    if shot_summary.get("support_balanced") is not True:
        raise ValueError("adapter input preflight report support cache is not balanced")
    if shot_summary.get("shot") is not None and int(shot_summary["shot"]) != shot:
        raise ValueError(f"adapter input preflight shot mismatch, expected {shot}, found {shot_summary.get('shot')}")
    method_summary = report.get("per_method_input_summary", {}).get("rs_cpc", {})
    per_m_summary = find_split_summary(method_summary.get("per_shot"), shot_split)
    ready_by_m = per_m_summary.get("method_input_ready_by_M") if isinstance(per_m_summary, dict) else None
    if isinstance(ready_by_m, dict) and ready_by_m.get(str(m_value)) is not True:
        raise ValueError("adapter input preflight report is not ready for requested RS-CPC M")


def validate_rs_cpc_adapter_input_plan(
    *,
    adapter_input_plan: str | Path | None,
    dataset: str,
    backbone: str,
    shot: int,
    m_value: int,
    shot_split: str,
) -> None:
    if not adapter_input_plan:
        raise ValueError("cached RS-CPC evaluation requires --adapter-input-plan")
    plan = read_json(adapter_input_plan)
    if plan.get("source_preflight_is_valid") is False:
        raise ValueError("adapter input plan is not ready because its source preflight is invalid")
    if plan.get("dataset") != dataset:
        raise ValueError(f"adapter input plan dataset mismatch, expected {dataset}, found {plan.get('dataset')}")
    if plan.get("backbone") != backbone:
        raise ValueError(f"adapter input plan backbone mismatch, expected {backbone}, found {plan.get('backbone')}")
    row = find_rs_cpc_plan_row(plan.get("rows"), shot=shot, m_value=m_value, shot_split=shot_split)
    if not row or row.get("is_ready") is not True:
        raise ValueError("adapter input plan is not ready for requested RS-CPC shot and M")


def validate_rs_cpc_prototype_preflight(
    *,
    prototype_preflight_report: str | Path | None,
    dataset: str,
    backbone: str,
    shot: int,
    m_value: int,
    shot_split: str,
    prototype_init: str,
) -> None:
    if not prototype_preflight_report:
        raise ValueError("cached RS-CPC evaluation requires --prototype-preflight-report")
    report = read_json(prototype_preflight_report)
    if report.get("is_valid") is not True:
        raise ValueError("RS-CPC prototype preflight report is not ready")
    if report.get("dataset") != dataset:
        raise ValueError(f"RS-CPC prototype preflight dataset mismatch, expected {dataset}, found {report.get('dataset')}")
    if report.get("backbone") != backbone:
        raise ValueError(f"RS-CPC prototype preflight backbone mismatch, expected {backbone}, found {report.get('backbone')}")
    if not prototype_preflight_has_ready_combination(
        report.get("per_combination_summary"),
        shot=shot,
        m_value=m_value,
        shot_split=shot_split,
        prototype_init=prototype_init,
    ):
        raise ValueError("RS-CPC prototype preflight report is not ready for requested shot, M, and prototype_init")


def prototype_preflight_has_ready_combination(
    rows: Any, *, shot: int, m_value: int, shot_split: str, prototype_init: str
) -> bool:
    if not isinstance(rows, list):
        return False
    requested_tokens = split_tokens(shot_split)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("prototype_init") != prototype_init:
            continue
        if int_or_none(row.get("candidate_M")) != m_value:
            continue
        if row.get("shot") is not None and int(row["shot"]) != shot:
            continue
        if not (requested_tokens & split_tokens(str(row.get("shot_split", "")))):
            continue
        if row.get("is_ready") is True:
            return True
    return False


def find_rs_cpc_plan_row(rows: Any, *, shot: int, m_value: int, shot_split: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    requested_tokens = split_tokens(shot_split)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("method") != "rs_cpc":
            continue
        if int_or_none(row.get("candidate_M")) != m_value:
            continue
        if row.get("shot") is not None and int(row["shot"]) != shot:
            continue
        if requested_tokens & split_tokens(str(row.get("shot_split", ""))):
            return row
    return None


def create_rs_cpc_run_dir(
    output_dir: Path,
    *,
    dataset: str,
    backbone: str,
    shot: int,
    m_value: int,
    prototype_init: str,
    seed: int,
) -> tuple[str, Path]:
    base = output_dir / dataset / backbone / "rs_cpc" / f"shot_{shot}" / f"M_{m_value}" / prototype_init / f"seed_{seed}"
    for _ in range(100):
        run_id = create_run_id()
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique RS-CPC run directory under {base}")


def ensure_results_output_allowed(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "outputs" and parts[index + 1] == "preflight":
            raise ValueError("cached RS-CPC evaluation metrics must not be written under outputs/preflight")


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    main()
