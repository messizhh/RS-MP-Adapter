#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


SUPPORTED_METHODS = ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"]
RS_CPC_PROTOTYPE_INITS = ["mean", "random_group_mean", "medoid"]
RS_CPC_M_VALUES = [1, 2, 4, 8]
COMMAND_OUTPUT_DIR = "results/raw"
MATRIX_FIELDS = [
    "method",
    "seed",
    "shot",
    "M",
    "prototype_init",
    "manifest_path",
    "text_feature_cache_path",
    "adapter_plan_path",
    "adapter_preflight_path",
    "prototype_preflight_path",
    "output_dir",
    "run_mode",
    "is_paper_result",
    "requires_post_run_preflight",
    "command",
]
SAFETY_FLAGS = [
    "is_paper_result",
    "writes_results_raw",
    "computes_logits",
    "computes_accuracy",
    "evaluates_model",
    "trains_model",
    "modifies_results",
    "deletes_results",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a review-only server_full evaluation command pack.")
    parser.add_argument("--server-full-preflight-report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/server_full_command_packs")
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument("--paper-result-mode", choices=["candidate_only"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_server_full_evaluation_command_pack(
        server_full_preflight_report=args.server_full_preflight_report,
        dataset=args.dataset,
        backbone=args.backbone,
        seeds=args.seeds,
        shots=args.shots,
        methods=args.methods,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        paper_result_mode=args.paper_result_mode,
        command=shlex.join(sys.argv),
    )
    print(f"server_full_command_pack_dir={result['pack_dir']}")
    print(f"server_full_command_pack_json={result['json_path']}")
    print(f"server_full_command_pack_md={result['markdown_path']}")
    print(f"server_full_command_pack_sh={result['shell_path']}")
    print(f"server_full_expected_matrix_csv={result['matrix_csv_path']}")
    print(f"num_commands={result['num_commands']}")


def export_server_full_evaluation_command_pack(
    *,
    server_full_preflight_report: str | Path,
    dataset: str,
    backbone: str,
    seeds: list[int],
    shots: list[int],
    methods: list[str],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    paper_result_mode: str,
    command: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    if execution_env != "remote_server":
        raise ValueError("server_full evaluation command packs require --execution-env remote_server")
    if run_mode != "server_full":
        raise ValueError("server_full evaluation command packs require --run-mode server_full")
    if paper_result_mode != "candidate_only":
        raise ValueError("only --paper-result-mode candidate_only is supported")

    report_path = Path(server_full_preflight_report)
    report_bytes = report_path.read_bytes()
    report = read_json(report_path)
    validate_ready_report(report, dataset=dataset, backbone=backbone, seeds=seeds, shots=shots, methods=methods)

    command_rows = build_command_rows(
        report=report,
        dataset=dataset,
        backbone=backbone,
        seeds=seeds,
        shots=shots,
        methods=methods,
        execution_env=execution_env,
        run_mode=run_mode,
        paper_result_mode=paper_result_mode,
    )
    expected_num_runs = int(report.get("expected_num_runs") or 0)
    if len(command_rows) != expected_num_runs:
        raise ValueError(f"generated command count {len(command_rows)} does not match report expected_num_runs={expected_num_runs}")

    created_at = utc_now_iso()
    pack_dir = unique_pack_dir(output_root, dataset=dataset, backbone=backbone, seeds=seeds)
    json_payload = {
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
        "created_at": created_at,
        "git_commit": git_commit_hash(),
        "command": command or shlex.join(sys.argv),
        "source_script": "scripts/export_server_full_evaluation_command_pack.py",
        "source_server_full_preflight_report": str(report_path),
        "dataset": dataset,
        "backbone": backbone,
        "seeds": seeds,
        "shots": shots,
        "methods": methods,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "paper_result_mode": paper_result_mode,
        "review_only": True,
        "does_not_execute_evaluation": True,
        "do_not_run_blindly": True,
        "num_commands": len(command_rows),
        "command_rows": command_rows,
    }
    json_path = safe_write_json(pack_dir / "server_full_command_pack.json", json_payload)
    matrix_csv_path = safe_write_csv(
        pack_dir / "server_full_expected_matrix.csv",
        [csv_row(row) for row in command_rows],
        MATRIX_FIELDS,
    )
    markdown_path = write_text_no_overwrite(
        pack_dir / "server_full_command_pack.md",
        render_markdown_pack(
            source_report=report_path,
            dataset=dataset,
            backbone=backbone,
            seeds=seeds,
            shots=shots,
            methods=methods,
            created_at=created_at,
            command_rows=command_rows,
        ),
    )
    shell_path = write_text_no_overwrite(
        pack_dir / "server_full_command_pack.sh",
        render_shell_pack(dataset=dataset, backbone=backbone, seeds=seeds, command_rows=command_rows),
    )

    if report_path.read_bytes() != report_bytes:
        raise RuntimeError(f"input server_full preflight report was modified unexpectedly: {report_path}")
    return {
        "pack_dir": str(pack_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "shell_path": str(shell_path),
        "matrix_csv_path": str(matrix_csv_path),
        "num_commands": len(command_rows),
        "command_rows": command_rows,
    }


def validate_ready_report(
    report: dict[str, Any],
    *,
    dataset: str,
    backbone: str,
    seeds: list[int],
    shots: list[int],
    methods: list[str],
) -> None:
    if report.get("is_valid") is not True:
        raise ValueError("server_full protocol preflight report is not valid")
    if report.get("is_ready_for_server_full") is not True:
        raise ValueError("server_full protocol preflight report is not ready for server_full")
    if report.get("dataset") != dataset:
        raise ValueError(f"server_full report dataset mismatch, expected {dataset}, found {report.get('dataset')}")
    if report.get("backbone") != backbone:
        raise ValueError(f"server_full report backbone mismatch, expected {backbone}, found {report.get('backbone')}")
    if sorted(int(item) for item in report.get("seeds", [])) != sorted(seeds):
        raise ValueError(f"server_full report seeds do not match requested seeds={seeds}")
    if sorted(int(item) for item in report.get("shots", [])) != sorted(shots):
        raise ValueError(f"server_full report shots do not match requested shots={shots}")
    normalized_methods = list(dict.fromkeys(methods))
    unsupported = [method for method in normalized_methods if method not in SUPPORTED_METHODS]
    if unsupported:
        raise ValueError(f"unsupported methods requested: {unsupported}")
    if report.get("methods") != normalized_methods:
        raise ValueError(f"server_full report methods do not match requested methods={normalized_methods}")
    if report.get("errors"):
        raise ValueError("server_full protocol preflight report contains errors")
    expected = int(report.get("expected_num_runs") or 0)
    ready = int(report.get("ready_num_runs") or 0)
    if expected <= 0 or ready != expected:
        raise ValueError(f"server_full report is not fully ready: ready={ready}, expected={expected}")
    for flag in SAFETY_FLAGS:
        if bool(report.get(flag)):
            raise ValueError(f"server_full protocol preflight report has unsafe flag {flag}=true")


def build_command_rows(
    *,
    report: dict[str, Any],
    dataset: str,
    backbone: str,
    seeds: list[int],
    shots: list[int],
    methods: list[str],
    execution_env: str,
    run_mode: str,
    paper_result_mode: str,
) -> list[dict[str, Any]]:
    report_rows = index_report_rows(report.get("expected_run_matrix", []))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        artifacts = seed_artifacts(report, seed)
        for method in methods:
            if method == "zero_shot":
                source_row = require_report_row(report_rows, method="zero_shot", seed=seed, shot=None, m_value=None, prototype_init="")
                rows.append(
                    make_command_row(
                        dataset=dataset,
                        backbone=backbone,
                        method="zero_shot",
                        seed=seed,
                        shot=None,
                        m_value=None,
                        prototype_init="",
                        artifacts=artifacts,
                        source_row=source_row,
                        execution_env=execution_env,
                        run_mode=run_mode,
                        paper_result_mode=paper_result_mode,
                    )
                )
            elif method in {"tip_adapter", "proto_adapter"}:
                for shot in shots:
                    source_row = require_report_row(report_rows, method=method, seed=seed, shot=shot, m_value=None, prototype_init="")
                    rows.append(
                        make_command_row(
                            dataset=dataset,
                            backbone=backbone,
                            method=method,
                            seed=seed,
                            shot=shot,
                            m_value=None,
                            prototype_init="",
                            artifacts=artifacts,
                            source_row=source_row,
                            execution_env=execution_env,
                            run_mode=run_mode,
                            paper_result_mode=paper_result_mode,
                        )
                    )
            elif method == "rs_cpc":
                for combo in expected_rs_cpc_combinations(shots):
                    source_row = require_report_row(
                        report_rows,
                        method="rs_cpc",
                        seed=seed,
                        shot=combo["shot"],
                        m_value=combo["M"],
                        prototype_init=combo["prototype_init"],
                    )
                    rows.append(
                        make_command_row(
                            dataset=dataset,
                            backbone=backbone,
                            method="rs_cpc",
                            seed=seed,
                            shot=combo["shot"],
                            m_value=combo["M"],
                            prototype_init=combo["prototype_init"],
                            artifacts=artifacts,
                            source_row=source_row,
                            execution_env=execution_env,
                            run_mode=run_mode,
                            paper_result_mode=paper_result_mode,
                        )
                    )
    return rows


def make_command_row(
    *,
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str,
    artifacts: dict[str, Any],
    source_row: dict[str, Any],
    execution_env: str,
    run_mode: str,
    paper_result_mode: str,
) -> dict[str, Any]:
    if source_row.get("is_ready") is not True:
        raise ValueError(f"source run row is not ready: {source_row}")
    adapter_plan_path = artifacts["adapter_plan_path"] if method in {"tip_adapter", "proto_adapter", "rs_cpc"} else None
    adapter_preflight_path = artifacts["adapter_preflight_path"] if method in {"tip_adapter", "proto_adapter", "rs_cpc"} else None
    prototype_preflight_path = artifacts["prototype_preflight_path"] if method == "rs_cpc" else None
    row = {
        "method": method,
        "seed": seed,
        "shot": shot,
        "M": m_value,
        "prototype_init": prototype_init,
        "manifest_path": artifacts["manifest_path"],
        "text_feature_cache_path": artifacts["text_feature_cache_path"],
        "adapter_plan_path": adapter_plan_path,
        "adapter_preflight_path": adapter_preflight_path,
        "prototype_preflight_path": prototype_preflight_path,
        "output_dir": COMMAND_OUTPUT_DIR,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "paper_result_mode": paper_result_mode,
        "is_paper_result": False,
        "requires_post_run_preflight": True,
    }
    row["command"] = render_evaluation_command(dataset=dataset, backbone=backbone, row=row)
    return row


def render_evaluation_command(*, dataset: str, backbone: str, row: dict[str, Any]) -> str:
    method = row["method"]
    common_args: list[tuple[str, Any]] = [
        ("--dataset", dataset),
        ("--backbone", backbone),
        ("--method", method),
        ("--seed", row["seed"]),
        ("--base-split", f"base_seed{row['seed']}"),
        ("--manifest", row["manifest_path"]),
        ("--text-feature-cache", row["text_feature_cache_path"]),
        ("--eval-splits", ["val", "test"]),
        ("--output-dir", row["output_dir"]),
        ("--device", "cuda"),
        ("--execution-env", row["execution_env"]),
        ("--run-mode", row["run_mode"]),
    ]
    if method == "zero_shot":
        return render_python_command(
            "scripts/run_zero_shot.py",
            [("--config", "configs/methods/zero_shot_clip.yaml"), *common_args, ("--skip-preflight-check", True)],
        )
    if method == "tip_adapter":
        return render_python_command(
            "scripts/run_tip_adapter.py",
            [
                ("--config", "configs/methods/tip_adapter.yaml"),
                *common_args,
                ("--shot", row["shot"]),
                ("--shot-split", f"shot_{row['shot']}_seed{row['seed']}"),
                ("--adapter-input-plan", row["adapter_plan_path"]),
                ("--preflight-report", row["adapter_preflight_path"]),
            ],
        )
    if method == "proto_adapter":
        return render_python_command(
            "scripts/run_proto_adapter.py",
            [
                ("--config", "configs/methods/proto_adapter.yaml"),
                *common_args,
                ("--shot", row["shot"]),
                ("--shot-split", f"shot_{row['shot']}_seed{row['seed']}"),
                ("--adapter-input-plan", row["adapter_plan_path"]),
                ("--preflight-report", row["adapter_preflight_path"]),
            ],
        )
    if method == "rs_cpc":
        return render_python_command(
            "scripts/run_rs_cpc.py",
            [
                ("--config", "configs/methods/rs_cpc.yaml"),
                *common_args,
                ("--shot", row["shot"]),
                ("--shot-split", f"shot_{row['shot']}_seed{row['seed']}"),
                ("--M", row["M"]),
                ("--prototype-init", row["prototype_init"]),
                ("--adapter-input-plan", row["adapter_plan_path"]),
                ("--preflight-report", row["adapter_preflight_path"]),
                ("--prototype-preflight-report", row["prototype_preflight_path"]),
            ],
        )
    raise ValueError(f"unsupported method: {method}")


def render_python_command(script_path: str, args: list[tuple[str, Any]]) -> str:
    rendered = [f"python3 {script_path} \\"]
    parts: list[str] = []
    for flag, value in args:
        if value is False or value is None or value == "":
            continue
        if value is True:
            parts.append(f"  {flag}")
        elif isinstance(value, list):
            joined = " ".join(shlex.quote(str(item)) for item in value)
            parts.append(f"  {flag} {joined}")
        else:
            parts.append(f"  {flag} {shlex.quote(str(value))}")
    for index, part in enumerate(parts):
        suffix = " \\" if index < len(parts) - 1 else ""
        rendered.append(f"{part}{suffix}")
    return "\n".join(rendered)


def index_report_rows(rows: Any) -> dict[tuple[str, int, int | None, int | None, str], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("server_full report expected_run_matrix must be a list")
    result: dict[tuple[str, int, int | None, int | None, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row_key(
            method=str(row.get("method") or ""),
            seed=int_or_none(row.get("seed")),
            shot=int_or_none(row.get("shot")),
            m_value=int_or_none(row.get("M")),
            prototype_init=str(row.get("prototype_init") or ""),
        )
        if key in result:
            raise ValueError(f"duplicate server_full expected_run_matrix row: {key}")
        result[key] = row
    return result


def require_report_row(
    rows: dict[tuple[str, int, int | None, int | None, str], dict[str, Any]],
    *,
    method: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str,
) -> dict[str, Any]:
    key = row_key(method=method, seed=seed, shot=shot, m_value=m_value, prototype_init=prototype_init)
    if key not in rows:
        raise ValueError(f"server_full report is missing expected run row: {key}")
    return rows[key]


def row_key(
    *,
    method: str,
    seed: int | None,
    shot: int | None,
    m_value: int | None,
    prototype_init: str,
) -> tuple[str, int, int | None, int | None, str]:
    if seed is None:
        raise ValueError("run row seed is missing")
    return (method, seed, shot, m_value, prototype_init)


def seed_artifacts(report: dict[str, Any], seed: int) -> dict[str, str]:
    summaries = report.get("seed_artifact_summary")
    if not isinstance(summaries, dict):
        raise ValueError("server_full report is missing seed_artifact_summary")
    status = summaries.get(str(seed)) or summaries.get(seed)
    if not isinstance(status, dict):
        raise ValueError(f"server_full report is missing artifact summary for seed {seed}")
    text_status = status.get("text_cache_status") if isinstance(status.get("text_cache_status"), dict) else {}
    adapter_plan_status = status.get("adapter_plan_status") if isinstance(status.get("adapter_plan_status"), dict) else {}
    adapter_preflight_status = (
        status.get("adapter_preflight_status") if isinstance(status.get("adapter_preflight_status"), dict) else {}
    )
    prototype_preflight_status = (
        status.get("prototype_preflight_status") if isinstance(status.get("prototype_preflight_status"), dict) else {}
    )
    artifacts = {
        "manifest_path": required_path(status.get("manifest_path"), f"seed{seed} manifest_path"),
        "text_feature_cache_path": required_path(
            text_status.get("selected_text_cache_path"),
            f"seed{seed} text_cache_status.selected_text_cache_path",
        ),
        "adapter_plan_path": required_path(adapter_plan_status.get("path"), f"seed{seed} adapter_plan_status.path"),
        "adapter_preflight_path": required_path(
            adapter_preflight_status.get("path"),
            f"seed{seed} adapter_preflight_status.path",
        ),
        "prototype_preflight_path": required_path(
            prototype_preflight_status.get("path"),
            f"seed{seed} prototype_preflight_status.path",
        ),
    }
    return artifacts


def required_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"server_full report is missing {context}")
    return value


def expected_rs_cpc_combinations(shots: list[int]) -> list[dict[str, Any]]:
    combinations: list[dict[str, Any]] = []
    for shot in shots:
        for init_mode in RS_CPC_PROTOTYPE_INITS:
            for m_value in RS_CPC_M_VALUES:
                if init_mode == "mean" and m_value != 1:
                    continue
                if m_value > shot:
                    continue
                combinations.append({"shot": shot, "M": m_value, "prototype_init": init_mode})
    return combinations


def render_markdown_pack(
    *,
    source_report: Path,
    dataset: str,
    backbone: str,
    seeds: list[int],
    shots: list[int],
    methods: list[str],
    created_at: str,
    command_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Server Full Evaluation Command Pack",
        "",
        "This command pack does not execute evaluation.",
        "server_full is a protocol/run mode, not automatically a paper result.",
        "Paper-facing results require explicit allow flag and post-run result preflight.",
        "",
        "The generated shell file is review-only and exits before the command block.",
        "No `--allow-paper-result` flag is added by this exporter.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Backbone: `{backbone}`",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Shots: `{', '.join(str(shot) for shot in shots)}`",
        f"- Methods: `{', '.join(methods)}`",
        f"- Source server_full preflight report: `{source_report}`",
        f"- Created at: `{created_at}`",
        f"- Number of commands: `{len(command_rows)}`",
        "",
        "## Matrix Summary",
        "",
    ]
    for method in SUPPORTED_METHODS:
        count = sum(1 for row in command_rows if row["method"] == method)
        if count:
            lines.append(f"- `{method}`: {count}")
    lines.extend(["", "## Command Preview", ""])
    for row in command_rows[:8]:
        label = command_label(row)
        lines.extend([f"### {label}", "", "```bash", row["command"], "```", ""])
    if len(command_rows) > 8:
        lines.append(f"... {len(command_rows) - 8} additional commands are listed in the JSON, CSV, and shell pack.")
    lines.append("")
    return "\n".join(lines)


def render_shell_pack(*, dataset: str, backbone: str, seeds: list[int], command_rows: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'echo "This command pack is generated for review. Do not run blindly."',
        "exit 1",
        "",
        ": <<'SERVER_FULL_COMMAND_PACK'",
        f"# Review-only server_full evaluation commands for {dataset}/{backbone}/seeds {','.join(str(seed) for seed in seeds)}",
        "# This command pack does not execute evaluation.",
        "# server_full is a protocol/run mode, not automatically a paper result.",
        "# Paper-facing results require explicit allow flag and post-run result preflight.",
        "",
    ]
    for row in command_rows:
        lines.extend([f"# {command_label(row)}", row["command"], ""])
    lines.append("SERVER_FULL_COMMAND_PACK")
    lines.append("")
    return "\n".join(lines)


def command_label(row: dict[str, Any]) -> str:
    parts = [str(row["method"]), f"seed={row['seed']}"]
    if row.get("shot") is not None:
        parts.append(f"shot={row['shot']}")
    if row.get("M") is not None:
        parts.append(f"M={row['M']}")
    if row.get("prototype_init"):
        parts.append(f"prototype_init={row['prototype_init']}")
    return " ".join(parts)


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "seed": row["seed"],
        "shot": "" if row["shot"] is None else row["shot"],
        "M": "" if row["M"] is None else row["M"],
        "prototype_init": row["prototype_init"],
        "manifest_path": row["manifest_path"],
        "text_feature_cache_path": row["text_feature_cache_path"],
        "adapter_plan_path": row["adapter_plan_path"] or "",
        "adapter_preflight_path": row["adapter_preflight_path"] or "",
        "prototype_preflight_path": row["prototype_preflight_path"] or "",
        "output_dir": row["output_dir"],
        "run_mode": row["run_mode"],
        "is_paper_result": row["is_paper_result"],
        "requires_post_run_preflight": row["requires_post_run_preflight"],
        "command": row["command"],
    }


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def unique_pack_dir(output_dir: Path, *, dataset: str, backbone: str, seeds: list[int]) -> Path:
    seed_part = "_".join(f"seed{seed}" for seed in seeds)
    base_dir = output_dir / f"{dataset}_{backbone}_{seed_part}"
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique server_full command pack directory under {base_dir}")


def write_text_no_overwrite(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8")
    return path


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("server_full evaluation command packs must not be written under results/raw")


if __name__ == "__main__":
    main()
