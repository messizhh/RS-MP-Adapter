from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.config_loader import save_config_snapshot
from src.logging.system_info import get_system_info
from src.utils.io import write_json_no_overwrite
from src.utils.timing import utc_now_iso


LOCAL_RUN_MODES = {"dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"}


@dataclass
class ExperimentRun:
    run_id: str
    run_dir: Path
    config_snapshot_path: Path
    metadata_path: Path
    metrics_path: Path
    log_path: Path
    start_time: str


def create_run_id() -> str:
    return f"{utc_now_iso().replace(':', '').replace('-', '').split('.')[0]}_{uuid.uuid4().hex[:8]}"


def create_unique_run_dir(
    output_dir: str | Path,
    dataset: str,
    backbone: str,
    method: str,
    shot: int | None,
    seed: int,
) -> tuple[str, Path]:
    shot_name = f"shot_{shot}" if shot is not None else "shot_none"
    base = Path(output_dir) / dataset / backbone / method / shot_name / f"seed_{seed}"
    for _ in range(100):
        run_id = create_run_id()
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {base}")


def is_paper_allowed(execution_env: str, run_mode: str, requested: bool) -> bool:
    if not requested:
        return False
    if execution_env == "local_wsl" or run_mode in LOCAL_RUN_MODES:
        return False
    return True


def start_experiment_run(
    output_dir: str | Path,
    config: dict[str, Any],
    config_path: str | Path | None,
    dataset: str,
    backbone: str,
    method: str,
    shot: int | None,
    seed: int,
    execution_env: str,
    run_mode: str,
    device: str,
    split_path: str | Path | None = None,
    server_job_id: str | None = None,
    is_paper_result: bool = False,
) -> tuple[ExperimentRun, dict[str, Any]]:
    run_id, run_dir = create_unique_run_dir(output_dir, dataset, backbone, method, shot, seed)
    config_snapshot_path = save_config_snapshot(config, run_dir)
    log_path = run_dir / "log.txt"
    log_path.write_text("Experiment log initialized.\n", encoding="utf-8")
    start_time = utc_now_iso()
    metadata = {
        **get_system_info(device=device),
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "config_path": str(config_path) if config_path is not None else "",
        "config_snapshot_path": str(config_snapshot_path),
        "seed": seed,
        "dataset": dataset,
        "shot": shot,
        "backbone": backbone,
        "method": method,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": is_paper_allowed(execution_env, run_mode, is_paper_result),
        "device": device,
        "server_job_id": server_job_id,
        "split_path": str(split_path) if split_path is not None else "",
        "start_time": start_time,
        "end_time": "",
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(log_path),
    }
    run = ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        metadata_path=run_dir / "metadata.json",
        metrics_path=run_dir / "metrics.json",
        log_path=log_path,
        start_time=start_time,
    )
    return run, metadata


def finish_experiment_run(
    run: ExperimentRun,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    end_time = utc_now_iso()
    metadata = dict(metadata)
    metadata["end_time"] = end_time
    metrics = dict(metrics)
    metrics.setdefault("run_id", run.run_id)
    metrics.setdefault("dataset", metadata["dataset"])
    metrics.setdefault("shot", metadata["shot"])
    metrics.setdefault("backbone", metadata["backbone"])
    metrics.setdefault("method", metadata["method"])
    metrics.setdefault("seed", metadata["seed"])
    metrics.setdefault("device", metadata["device"])
    metrics.setdefault("execution_env", metadata["execution_env"])
    metrics.setdefault("run_mode", metadata["run_mode"])
    metrics.setdefault("is_paper_result", metadata["is_paper_result"])
    metrics.setdefault("config_path", metadata["config_path"])
    metrics.setdefault("config_snapshot_path", metadata["config_snapshot_path"])
    metrics.setdefault("split_path", metadata["split_path"])
    metrics.setdefault("result_json_path", str(run.metrics_path))
    metrics.setdefault("log_path", str(run.log_path))
    metrics.setdefault("start_time", metadata["start_time"])
    metrics.setdefault("end_time", end_time)
    metrics_path = write_json_no_overwrite(run.metrics_path, metrics)
    metadata_path = write_json_no_overwrite(run.metadata_path, metadata)
    with run.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"Experiment finished at {end_time}.\n")
    return metadata_path, metrics_path
