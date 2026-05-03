from __future__ import annotations

from typing import Any


REQUIRED_METRICS_FIELDS = {
    "run_id",
    "method",
    "backbone",
    "dataset",
    "shot",
    "seed",
    "execution_env",
    "run_mode",
    "is_paper_result",
    "device",
    "top1_acc",
    "cache_entries",
    "trainable_params",
    "training_time_sec",
    "inference_time_sec",
    "images_per_second",
    "gpu_memory_mb",
    "config_path",
    "config_snapshot_path",
    "split_path",
    "result_json_path",
    "log_path",
    "start_time",
    "end_time",
    "uses_fake_data",
    "uses_fake_features",
    "fake_or_dry_run",
}

REQUIRED_METADATA_FIELDS = {
    "run_id",
    "git_commit",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "host_name",
    "command",
    "config_path",
    "config_snapshot_path",
    "seed",
    "dataset",
    "shot",
    "backbone",
    "method",
    "execution_env",
    "run_mode",
    "is_paper_result",
    "device",
    "server_job_id",
    "split_path",
    "start_time",
    "end_time",
    "result_json_path",
    "log_path",
}


def missing_metrics_fields(metrics: dict[str, Any]) -> list[str]:
    return sorted(field for field in REQUIRED_METRICS_FIELDS if field not in metrics)


def missing_metadata_fields(metadata: dict[str, Any]) -> list[str]:
    return sorted(field for field in REQUIRED_METADATA_FIELDS if field not in metadata)


def validate_metrics_schema(metrics: dict[str, Any]) -> None:
    missing = missing_metrics_fields(metrics)
    if missing:
        raise ValueError(f"metrics.json missing required fields: {', '.join(missing)}")


def validate_metadata_schema(metadata: dict[str, Any]) -> None:
    missing = missing_metadata_fields(metadata)
    if missing:
        raise ValueError(f"metadata.json missing required fields: {', '.join(missing)}")
