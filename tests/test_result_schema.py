from __future__ import annotations

import unittest

from src.logging.result_schema import (
    REQUIRED_METADATA_FIELDS,
    REQUIRED_METRICS_FIELDS,
    missing_metadata_fields,
    missing_metrics_fields,
    validate_metadata_schema,
    validate_metrics_schema,
)


class ResultSchemaTest(unittest.TestCase):
    def test_metrics_schema_requires_phase1f_fields(self) -> None:
        metrics = {field: "" for field in REQUIRED_METRICS_FIELDS}
        metrics.update(
            {
                "shot": 1,
                "seed": 1,
                "is_paper_result": False,
                "top1_acc": 0.0,
                "cache_entries": 0,
                "trainable_params": 0,
                "training_time_sec": 0.0,
                "inference_time_sec": 0.0,
                "images_per_second": 0.0,
                "gpu_memory_mb": None,
                "uses_fake_data": True,
                "uses_fake_features": True,
                "fake_or_dry_run": True,
            }
        )
        validate_metrics_schema(metrics)
        for required in [
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
        ]:
            self.assertIn(required, REQUIRED_METRICS_FIELDS)

    def test_metrics_schema_reports_missing_fields(self) -> None:
        missing = missing_metrics_fields({"run_id": "abc"})
        self.assertIn("method", missing)
        with self.assertRaisesRegex(ValueError, "metrics.json missing required fields"):
            validate_metrics_schema({"run_id": "abc"})

    def test_metadata_schema_requires_runtime_metadata_fields(self) -> None:
        metadata = {field: "" for field in REQUIRED_METADATA_FIELDS}
        metadata.update({"shot": 1, "seed": 1, "is_paper_result": False, "server_job_id": None})
        validate_metadata_schema(metadata)
        for required in [
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
        ]:
            self.assertIn(required, REQUIRED_METADATA_FIELDS)

    def test_metadata_schema_reports_missing_fields(self) -> None:
        missing = missing_metadata_fields({"run_id": "abc"})
        self.assertIn("git_commit", missing)
        with self.assertRaisesRegex(ValueError, "metadata.json missing required fields"):
            validate_metadata_schema({"run_id": "abc"})


if __name__ == "__main__":
    unittest.main()
