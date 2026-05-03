#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local fake end-to-end Phase 1 pipeline.")
    parser.add_argument("--config", default="configs/methods/zero_shot_clip.yaml")
    parser.add_argument("--dataset", default="eurosat")
    parser.add_argument("--backbone", default="fake_backbone")
    parser.add_argument("--method", default="fake_pipeline")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="")
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execution_env != "local_wsl" or args.run_mode != "smoke_test" or args.device != "cpu":
        raise SystemExit("Fake pipeline must run as --execution-env local_wsl --run-mode smoke_test --device cpu.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_smoke_test.py",
            "--dry-run",
            "--run-mode",
            args.run_mode,
            "--execution-env",
            args.execution_env,
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = parse_stdout_paths(completed.stdout)
    metadata = read_json(paths["metadata_path"])
    metrics = read_json(paths["metrics_path"])
    table_summary = read_json(paths["table_summary_path"])
    method_metrics = {name: read_json(Path(path)) for name, path in paths.items() if name.endswith("_metrics_path") and name != "metrics_path"}

    if metadata["execution_env"] != "local_wsl" or metadata["run_mode"] != "smoke_test" or metadata["device"] != "cpu":
        raise RuntimeError("Fake pipeline metadata lost local smoke guardrails.")
    if metadata["is_paper_result"] or metrics["is_paper_result"]:
        raise RuntimeError("Fake pipeline output was incorrectly marked as a paper result.")
    if not metrics["uses_fake_data"] or not metrics["uses_fake_features"] or not metrics["fake_or_dry_run"]:
        raise RuntimeError("Fake pipeline metrics are missing fake-data guardrails.")
    for method_name, method_payload in method_metrics.items():
        if method_payload["is_paper_result"] or not method_payload["fake_or_dry_run"]:
            raise RuntimeError(f"{method_name} did not remain a non-paper fake/local run.")
    if table_summary["num_eligible_results"] != 0:
        raise RuntimeError("Default export did not exclude fake smoke results.")

    summary = {
        "created_at": utc_now_iso(),
        "source_script": "scripts/run_fake_pipeline.py",
        "execution_env": args.execution_env,
        "run_mode": args.run_mode,
        "device": args.device,
        "is_paper_result": False,
        "uses_fake_data": True,
        "uses_fake_features": True,
        "fake_or_dry_run": True,
        "metadata_path": str(paths["metadata_path"]),
        "metrics_path": str(paths["metrics_path"]),
        "feature_cache_path": str(paths["feature_cache_path"]),
        "extracted_feature_cache_path": str(paths["extracted_feature_cache_path"]),
        "feature_cache_validation_report_path": str(paths["feature_cache_validation_report_path"]),
        "dataset_inspection_report_path": str(paths["dataset_inspection_report_path"]),
        "split_path": str(paths["split_path"]),
        "method_metrics_paths": {name.removesuffix("_metrics_path"): str(path) for name, path in paths.items() if name.endswith("_metrics_path") and name != "metrics_path"},
        "table_summary_path": str(paths["table_summary_path"]),
        "num_eligible_table_results": table_summary["num_eligible_results"],
    }
    summary_path = safe_write_json(unique_path(output_dir / "fake_pipeline_summary.json"), summary)
    print(completed.stdout, end="")
    print(f"fake_pipeline_summary_path={summary_path}")


def parse_stdout_paths(stdout: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith("_path") or key.endswith("_dir"):
            paths[key] = Path(value)
    required = {
        "metadata_path",
        "metrics_path",
        "feature_cache_path",
        "extracted_feature_cache_path",
        "feature_cache_validation_report_path",
        "dataset_inspection_report_path",
        "split_path",
        "table_summary_path",
        "linear_probe_metrics_path",
        "tip_adapter_metrics_path",
        "proto_adapter_metrics_path",
        "rs_cpc_metrics_path",
    }
    missing = sorted(required - set(paths))
    if missing:
        raise RuntimeError(f"Fake pipeline missing expected output paths: {', '.join(missing)}")
    return paths


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find non-existing output path for {path}")


if __name__ == "__main__":
    main()
