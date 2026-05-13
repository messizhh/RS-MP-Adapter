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


PHASES = [
    ("A", "dataset/split readiness"),
    ("B", "image feature cache manifest and support caches"),
    ("C", "standalone text feature cache"),
    ("D", "adapter input preflight"),
    ("E", "adapter input plan"),
    ("F", "RS-CPC prototype preflight"),
    ("G", "rerun server_full protocol preflight"),
]
PLAN_FIELDS = [
    "seed",
    "phase",
    "artifact_type",
    "expected_path_or_pattern",
    "current_status",
    "blocking_reason_from_report",
    "suggested_script",
    "suggested_command_template",
    "is_paper_result",
    "writes_results_raw",
    "computes_logits",
    "computes_accuracy",
    "trains_model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a read-only seed expansion plan from server_full protocol preflight.")
    parser.add_argument("--server-full-preflight-report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--target-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/seed_expansion_plans")
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_seed_expansion_plan(
        server_full_preflight_report=args.server_full_preflight_report,
        dataset=args.dataset,
        backbone=args.backbone,
        target_seeds=args.target_seeds,
        shots=args.shots,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        command=shlex.join(sys.argv),
    )
    print(f"seed_expansion_plan_dir={result['plan_dir']}")
    print(f"seed_expansion_plan_json={result['json_path']}")
    print(f"seed_expansion_plan_csv={result['csv_path']}")
    print(f"seed_expansion_plan_md={result['markdown_path']}")
    print(f"num_plan_items={result['num_plan_items']}")


def export_seed_expansion_plan(
    *,
    server_full_preflight_report: str | Path,
    dataset: str,
    backbone: str,
    target_seeds: list[int],
    shots: list[int],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> dict[str, Any]:
    report_path = Path(server_full_preflight_report)
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    report_bytes = report_path.read_bytes()
    report = read_json(report_path)
    validate_report_context(report, dataset=dataset, backbone=backbone)

    plan_items: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    for seed in target_seeds:
        seed_status = seed_status_from_report(report, seed)
        seed_summaries[str(seed)] = summarize_seed_status(seed_status)
        plan_items.extend(build_seed_plan_items(dataset=dataset, backbone=backbone, seed=seed, shots=shots, seed_status=seed_status))

    created_at = utc_now_iso()
    plan_dir = unique_plan_dir(output_root, dataset=dataset, backbone=backbone, target_seeds=target_seeds)
    csv_path = safe_write_csv(plan_dir / "seed_expansion_plan.csv", [csv_row(item) for item in plan_items], PLAN_FIELDS)
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
        "source_script": "scripts/export_seed_expansion_plan.py",
        "source_server_full_preflight_report": str(report_path),
        "dataset": dataset,
        "backbone": backbone,
        "target_seeds": target_seeds,
        "shots": shots,
        "source_report_summary": {
            "is_valid": report.get("is_valid"),
            "is_ready_for_server_full": report.get("is_ready_for_server_full"),
            "expected_num_runs": report.get("expected_num_runs"),
            "ready_num_runs": report.get("ready_num_runs"),
            "seeds": report.get("seeds"),
        },
        "seed_summaries": seed_summaries,
        "phases": [{"phase": key, "name": name} for key, name in PHASES],
        "num_plan_items": len(plan_items),
        "plan_items": plan_items,
    }
    json_path = safe_write_json(plan_dir / "seed_expansion_plan.json", json_payload)
    markdown_path = write_text_no_overwrite(
        plan_dir / "seed_expansion_plan.md",
        render_markdown_plan(
            dataset=dataset,
            backbone=backbone,
            target_seeds=target_seeds,
            shots=shots,
            report_path=report_path,
            created_at=created_at,
            plan_items=plan_items,
        ),
    )

    if report_path.exists() and report_path.read_bytes() != report_bytes:
        raise RuntimeError(f"input report was modified unexpectedly: {report_path}")
    return {
        "plan_dir": str(plan_dir),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "num_plan_items": len(plan_items),
        "plan_items": plan_items,
    }


def build_seed_plan_items(
    *,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    seed_status: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        make_item(
            seed=seed,
            phase="A",
            artifact_type="dataset/split readiness",
            expected_path_or_pattern=split_pattern(dataset, seed, shots),
            current_status="unknown" if seed_status else "missing",
            blocking_reasons=phase_a_reasons(seed_status),
            suggested_script="scripts/generate_splits.py",
            suggested_command_template=split_command(dataset, seed, shots),
        ),
        make_item(
            seed=seed,
            phase="B",
            artifact_type="image feature cache manifest and support caches",
            expected_path_or_pattern=image_feature_pattern(dataset, backbone, seed, shots),
            current_status=ready_or_missing(image_phase_ready(seed_status, shots)),
            blocking_reasons=image_phase_reasons(seed_status, shots),
            suggested_script="scripts/extract_features.py; scripts/build_feature_cache_manifest.py",
            suggested_command_template=image_feature_command(dataset, backbone, seed, shots),
        ),
        make_item(
            seed=seed,
            phase="C",
            artifact_type="standalone text feature cache",
            expected_path_or_pattern=f"outputs/features/{backbone}/{dataset}/base_seed{seed}/{dataset}/{backbone}/text/*/text_feature_cache.pt",
            current_status=ready_or_missing(text_phase_ready(seed_status)),
            blocking_reasons=text_phase_reasons(seed_status),
            suggested_script="scripts/check_text_feature_cache_preflight.py; scripts/extract_text_features.py",
            suggested_command_template=text_feature_command(dataset, backbone, seed),
        ),
        make_item(
            seed=seed,
            phase="D",
            artifact_type="adapter input preflight",
            expected_path_or_pattern=f"outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json",
            current_status=ready_or_missing(json_status_ready(seed_status, "adapter_preflight_status")),
            blocking_reasons=json_status_reasons(seed_status, "adapter_preflight_status", "adapter_input_preflight_missing"),
            suggested_script="scripts/check_adapter_input_preflight.py",
            suggested_command_template=adapter_preflight_command(dataset, backbone, seed, shots),
        ),
        make_item(
            seed=seed,
            phase="E",
            artifact_type="adapter input plan",
            expected_path_or_pattern=f"outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/*/adapter_input_plan.json",
            current_status=ready_or_missing(json_status_ready(seed_status, "adapter_plan_status")),
            blocking_reasons=json_status_reasons(seed_status, "adapter_plan_status", "adapter_input_plan_missing"),
            suggested_script="scripts/export_adapter_input_plan.py",
            suggested_command_template=adapter_plan_command(dataset, backbone, seed),
        ),
        make_item(
            seed=seed,
            phase="F",
            artifact_type="RS-CPC prototype preflight",
            expected_path_or_pattern=f"outputs/preflight/rs_cpc_prototypes/{dataset}_{backbone}_seed{seed}/*/rs_cpc_prototype_preflight_report.json",
            current_status=ready_or_missing(json_status_ready(seed_status, "prototype_preflight_status")),
            blocking_reasons=json_status_reasons(seed_status, "prototype_preflight_status", "rs_cpc_prototype_preflight_missing"),
            suggested_script="scripts/check_rs_cpc_prototype_preflight.py",
            suggested_command_template=prototype_preflight_command(dataset, backbone, seed),
        ),
        make_item(
            seed=seed,
            phase="G",
            artifact_type="rerun server_full protocol preflight",
            expected_path_or_pattern=f"outputs/preflight/server_full_protocol/{dataset}_{backbone}/<TIMESTAMP>/server_full_protocol_preflight_report.json",
            current_status="unknown",
            blocking_reasons=["rerun_required_after_seed_artifacts_are_created"],
            suggested_script="scripts/check_server_full_protocol_preflight.py",
            suggested_command_template=server_full_preflight_command(dataset, backbone, shots),
        ),
    ]


def make_item(
    *,
    seed: int,
    phase: str,
    artifact_type: str,
    expected_path_or_pattern: str,
    current_status: str,
    blocking_reasons: list[str],
    suggested_script: str,
    suggested_command_template: str,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "phase": phase,
        "artifact_type": artifact_type,
        "expected_path_or_pattern": expected_path_or_pattern,
        "current_status": current_status,
        "blocking_reason_from_report": sorted(set(reason for reason in blocking_reasons if reason)),
        "suggested_script": suggested_script,
        "suggested_command_template": suggested_command_template,
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
    }


def phase_a_reasons(seed_status: dict[str, Any]) -> list[str]:
    if not seed_status:
        return ["seed_not_in_server_full_preflight_report", "split_readiness_not_checked_by_server_full_preflight"]
    return ["split_readiness_not_checked_by_server_full_preflight"]


def image_phase_ready(seed_status: dict[str, Any], shots: list[int]) -> bool:
    if not seed_status or not seed_status.get("manifest_exists"):
        return False
    base_cache_paths = seed_status.get("base_cache_paths") if isinstance(seed_status.get("base_cache_paths"), dict) else {}
    support_cache_paths = seed_status.get("support_cache_paths") if isinstance(seed_status.get("support_cache_paths"), dict) else {}
    return bool(base_cache_paths.get("val") and base_cache_paths.get("test") and all(support_cache_paths.get(str(shot)) or support_cache_paths.get(shot) for shot in shots))


def image_phase_reasons(seed_status: dict[str, Any], shots: list[int]) -> list[str]:
    reasons = seed_errors_matching(
        seed_status,
        prefixes=["missing_manifest", "manifest_", "missing_base_val_cache", "missing_base_test_cache"],
    )
    for shot in shots:
        reason = f"missing_support_cache_shot_{shot}"
        if reason in seed_errors(seed_status):
            reasons.append(reason)
    if not seed_status:
        reasons.append("seed_not_in_server_full_preflight_report")
    return reasons


def text_phase_ready(seed_status: dict[str, Any]) -> bool:
    status = seed_status.get("text_cache_status") if isinstance(seed_status, dict) else {}
    return bool(isinstance(status, dict) and status.get("is_ready") is True)


def text_phase_reasons(seed_status: dict[str, Any]) -> list[str]:
    status = seed_status.get("text_cache_status") if isinstance(seed_status, dict) else {}
    if isinstance(status, dict) and status.get("blocking_reason"):
        return [str(status["blocking_reason"])]
    return seed_errors_matching(seed_status, prefixes=["missing_text_cache", "text_cache_"]) or ["standalone_text_cache_status_unknown"]


def json_status_ready(seed_status: dict[str, Any], field: str) -> bool:
    status = seed_status.get(field) if isinstance(seed_status, dict) else {}
    return bool(isinstance(status, dict) and status.get("is_ready") is True)


def json_status_reasons(seed_status: dict[str, Any], field: str, default_missing: str) -> list[str]:
    status = seed_status.get(field) if isinstance(seed_status, dict) else {}
    if isinstance(status, dict):
        raw = status.get("blocking_reasons")
        if isinstance(raw, list) and raw:
            return [str(item) for item in raw]
        if status.get("blocking_reason"):
            return [str(status["blocking_reason"])]
    return seed_errors_matching(seed_status, prefixes=[default_missing, default_missing.replace("_missing", "_not_ready")]) or [default_missing]


def seed_errors(seed_status: dict[str, Any]) -> list[str]:
    if not isinstance(seed_status, dict):
        return []
    raw = seed_status.get("seed_errors")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def seed_errors_matching(seed_status: dict[str, Any], *, prefixes: list[str]) -> list[str]:
    return [reason for reason in seed_errors(seed_status) if any(reason.startswith(prefix) for prefix in prefixes)]


def ready_or_missing(is_ready: bool) -> str:
    return "ready" if is_ready else "missing"


def validate_report_context(report: dict[str, Any], *, dataset: str, backbone: str) -> None:
    if report.get("dataset") != dataset:
        raise ValueError(f"server_full preflight dataset mismatch, expected {dataset}, found {report.get('dataset')}")
    if report.get("backbone") != backbone:
        raise ValueError(f"server_full preflight backbone mismatch, expected {backbone}, found {report.get('backbone')}")


def seed_status_from_report(report: dict[str, Any], seed: int) -> dict[str, Any]:
    summary = report.get("seed_artifact_summary")
    if not isinstance(summary, dict):
        return {}
    value = summary.get(str(seed), summary.get(seed))
    return value if isinstance(value, dict) else {}


def summarize_seed_status(seed_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "present_in_source_report": bool(seed_status),
        "seed_errors": seed_errors(seed_status),
        "manifest_exists": seed_status.get("manifest_exists") if isinstance(seed_status, dict) else None,
        "text_cache_ready": text_phase_ready(seed_status),
        "adapter_preflight_ready": json_status_ready(seed_status, "adapter_preflight_status"),
        "adapter_plan_ready": json_status_ready(seed_status, "adapter_plan_status"),
        "prototype_preflight_ready": json_status_ready(seed_status, "prototype_preflight_status"),
    }


def split_pattern(dataset: str, seed: int, shots: list[int]) -> str:
    shot_paths = " ".join(f"splits/{dataset}/shot_{shot}_seed{seed}.json" for shot in shots)
    return f"splits/{dataset}/base_split_seed{seed}.json; splits/{dataset}/base_seed{seed}.json; {shot_paths}"


def image_feature_pattern(dataset: str, backbone: str, seed: int, shots: list[int]) -> str:
    shot_parts = " ".join(f"shot_{shot}_seed{seed}" for shot in shots)
    return (
        f"outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json; "
        f"outputs/features/{backbone}/{dataset}/base_seed{seed}/{{train,val,test}}/feature_cache.pt; "
        f"support splits: {shot_parts}"
    )


def split_command(dataset: str, seed: int, shots: list[int]) -> str:
    return (
        "python3 scripts/generate_splits.py "
        f"--config configs/datasets/{dataset}.yaml "
        f"--dataset {dataset} "
        "--dataset-root <DATASET_ROOT_TODO> "
        f"--shots {' '.join(str(shot) for shot in shots)} "
        f"--seeds {seed} "
        f"--output-dir splits/{dataset} "
        "--verify-exact-command-before-running"
    )


def image_feature_command(dataset: str, backbone: str, seed: int, shots: list[int]) -> str:
    shot_text = " ".join(str(shot) for shot in shots)
    return (
        "TODO verify exact server command before running. "
        f"Extract image features for {dataset}/{backbone}/seed{seed} base train/val/test and support shots {shot_text}; "
        "then run: "
        "python3 scripts/build_feature_cache_manifest.py "
        f"--feature-root outputs/features/{backbone}/{dataset}/base_seed{seed} "
        f"--output-dir outputs/manifests/feature_cache_after_seed{seed}_support"
    )


def text_feature_command(dataset: str, backbone: str, seed: int) -> str:
    return (
        "python3 scripts/extract_text_features.py "
        f"--dataset {dataset} "
        f"--backbone {backbone} "
        f"--base-split base_seed{seed} "
        "--preflight-report <TEXT_FEATURE_CACHE_PREFLIGHT_REPORT_TODO> "
        f"--backbone-config configs/backbones/{backbone}.yaml "
        "--method-config configs/methods/zero_shot_clip.yaml "
        "--weights-path <BACKBONE_WEIGHTS_PATH_TODO> "
        "--output-dir outputs/features "
        "--device cuda "
        "--execution-env remote_server "
        "--run-mode server_full "
        "--verify-exact-command-before-running"
    )


def adapter_preflight_command(dataset: str, backbone: str, seed: int, shots: list[int]) -> str:
    shot_splits = " ".join(f"shot_{shot}_seed{seed}" for shot in shots)
    return (
        "python3 scripts/check_adapter_input_preflight.py "
        f"--manifest outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json "
        f"--dataset {dataset} "
        f"--backbone {backbone} "
        f"--base-split base_seed{seed} "
        f"--shot-splits {shot_splits} "
        "--methods tip_adapter proto_adapter rs_cpc "
        "--output-dir outputs/preflight/adapter_input "
        "--execution-env remote_server "
        "--run-mode local_validation"
    )


def adapter_plan_command(dataset: str, backbone: str, seed: int) -> str:
    return (
        "python3 scripts/export_adapter_input_plan.py "
        f"--preflight-report outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json "
        "--output-dir outputs/preflight/adapter_input_plans "
        "--execution-env remote_server "
        "--run-mode local_validation"
    )


def prototype_preflight_command(dataset: str, backbone: str, seed: int) -> str:
    return (
        "python3 scripts/check_rs_cpc_prototype_preflight.py "
        f"--adapter-input-plan outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/<TIMESTAMP_TODO>/adapter_input_plan.json "
        f"--preflight-report outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json "
        "--prototype-inits mean random_group_mean medoid "
        "--output-dir outputs/preflight/rs_cpc_prototypes "
        "--execution-env remote_server "
        "--run-mode local_validation"
    )


def server_full_preflight_command(dataset: str, backbone: str, shots: list[int]) -> str:
    return (
        "python3 scripts/check_server_full_protocol_preflight.py "
        f"--dataset {dataset} "
        f"--backbone {backbone} "
        "--seeds 1 2 3 "
        f"--shots {' '.join(str(shot) for shot in shots)} "
        "--methods zero_shot tip_adapter proto_adapter rs_cpc "
        "--rs-cpc-prototype-inits mean random_group_mean medoid "
        "--rs-cpc-M-values 1 2 4 8 "
        "--manifest-template 'outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json' "
        "--text-cache-template 'outputs/features/{backbone}/{dataset}/base_seed{seed}/{dataset}/{backbone}/text/*/text_feature_cache.pt' "
        "--adapter-plan-template 'outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/*/adapter_input_plan.json' "
        "--adapter-preflight-template 'outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json' "
        "--prototype-preflight-template 'outputs/preflight/rs_cpc_prototypes/{dataset}_{backbone}_seed{seed}/*/rs_cpc_prototype_preflight_report.json' "
        "--output-dir outputs/preflight/server_full_protocol "
        "--execution-env remote_server "
        "--run-mode local_validation"
    )


def render_markdown_plan(
    *,
    dataset: str,
    backbone: str,
    target_seeds: list[int],
    shots: list[int],
    report_path: Path,
    created_at: str,
    plan_items: list[dict[str, Any]],
) -> str:
    lines = [
        "# Seed Expansion Plan",
        "",
        "This is a planning artifact only.",
        "It does not run experiments.",
        "It is not a paper result.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Backbone: `{backbone}`",
        f"- Target seeds: `{', '.join(str(seed) for seed in target_seeds)}`",
        f"- Shots: `{', '.join(str(shot) for shot in shots)}`",
        f"- Source server_full preflight report: `{report_path}`",
        f"- Created at: `{created_at}`",
        "",
        "| Seed | Phase | Artifact | Status | Blocking Reason | Suggested Script |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for item in plan_items:
        lines.append(
            "| {seed} | {phase} | {artifact_type} | {current_status} | {blocking} | {script} |".format(
                seed=item["seed"],
                phase=item["phase"],
                artifact_type=markdown_cell(item["artifact_type"]),
                current_status=item["current_status"],
                blocking=markdown_cell("; ".join(item["blocking_reason_from_report"])),
                script=markdown_cell(item["suggested_script"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def csv_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": item["seed"],
        "phase": item["phase"],
        "artifact_type": item["artifact_type"],
        "expected_path_or_pattern": item["expected_path_or_pattern"],
        "current_status": item["current_status"],
        "blocking_reason_from_report": ";".join(item["blocking_reason_from_report"]),
        "suggested_script": item["suggested_script"],
        "suggested_command_template": item["suggested_command_template"],
        "is_paper_result": item["is_paper_result"],
        "writes_results_raw": item["writes_results_raw"],
        "computes_logits": item["computes_logits"],
        "computes_accuracy": item["computes_accuracy"],
        "trains_model": item["trains_model"],
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def unique_plan_dir(output_dir: Path, *, dataset: str, backbone: str, target_seeds: list[int]) -> Path:
    seed_part = "_".join(f"seed{seed}" for seed in target_seeds)
    base_dir = output_dir / f"{dataset}_{backbone}_{seed_part}"
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique seed expansion plan directory under {base_dir}")


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
            raise ValueError("seed expansion plans must not be written under results/raw")


if __name__ == "__main__":
    main()
