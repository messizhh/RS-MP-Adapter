#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import pickle
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_csv, safe_write_json
from src.utils.timing import utc_now_iso


SUPPORTED_METHODS = {"zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"}
SUPPORTED_RS_CPC_INITS = {"mean", "random_group_mean", "medoid"}
MATRIX_FIELDS = [
    "method",
    "seed",
    "shot",
    "M",
    "prototype_init",
    "required_inputs",
    "is_ready",
    "blocking_reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only server_full protocol preflight before paper-facing runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--rs-cpc-prototype-inits", nargs="+", default=["mean", "random_group_mean", "medoid"])
    parser.add_argument("--rs-cpc-M-values", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--manifest-template", required=True)
    parser.add_argument("--text-cache-template", required=True)
    parser.add_argument("--adapter-plan-template", required=True)
    parser.add_argument("--adapter-preflight-template", required=True)
    parser.add_argument("--prototype-preflight-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, is_valid = run_server_full_protocol_preflight(
        dataset=args.dataset,
        backbone=args.backbone,
        seeds=args.seeds,
        shots=args.shots,
        methods=args.methods,
        rs_cpc_prototype_inits=args.rs_cpc_prototype_inits,
        rs_cpc_m_values=args.rs_cpc_M_values,
        manifest_template=args.manifest_template,
        text_cache_template=args.text_cache_template,
        adapter_plan_template=args.adapter_plan_template,
        adapter_preflight_template=args.adapter_preflight_template,
        prototype_preflight_template=args.prototype_preflight_template,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        command=shlex.join(sys.argv),
    )
    print(f"server_full_protocol_preflight_report_path={report_path}")
    print(f"is_valid={str(is_valid).lower()}")
    if not is_valid:
        raise SystemExit(1)


def run_server_full_protocol_preflight(
    *,
    dataset: str,
    backbone: str,
    seeds: list[int],
    shots: list[int],
    methods: list[str],
    rs_cpc_prototype_inits: list[str],
    rs_cpc_m_values: list[int],
    manifest_template: str,
    text_cache_template: str,
    adapter_plan_template: str,
    adapter_preflight_template: str,
    prototype_preflight_template: str,
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> tuple[Path, bool]:
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    errors: list[str] = []
    warnings: list[str] = []
    recommendations = [
        "server_full does not automatically imply is_paper_result=true.",
        "Formal evaluation commands must explicitly pass --allow-paper-result before a run can be marked as paper-facing.",
        "Do not promote local_validation outputs into paper-facing tables.",
    ]

    checked_methods = normalize_methods(methods, errors)
    prototype_inits, excluded_inits = normalize_rs_cpc_inits(rs_cpc_prototype_inits)
    for init_mode in excluded_inits:
        warnings.append(f"excluded unsupported RS-CPC prototype_init from server_full matrix: {init_mode}")
    legal_rs_cpc = legal_rs_cpc_combinations(
        shots=shots,
        prototype_inits=prototype_inits,
        m_values=rs_cpc_m_values,
    )

    seed_statuses: dict[int, dict[str, Any]] = {}
    expected_run_matrix: list[dict[str, Any]] = []
    for seed in seeds:
        status = inspect_seed_artifacts(
            dataset=dataset,
            backbone=backbone,
            seed=seed,
            shots=shots,
            methods=checked_methods,
            manifest_template=manifest_template,
            text_cache_template=text_cache_template,
            adapter_plan_template=adapter_plan_template,
            adapter_preflight_template=adapter_preflight_template,
            prototype_preflight_template=prototype_preflight_template,
        )
        seed_statuses[seed] = status
        errors.extend(f"seed{seed}: {reason}" for reason in status["seed_errors"])
        warnings.extend(f"seed{seed}: {warning}" for warning in status["warnings"])
        expected_run_matrix.extend(
            build_seed_run_rows(
                seed=seed,
                shots=shots,
                methods=checked_methods,
                legal_rs_cpc=legal_rs_cpc,
                status=status,
            )
        )

    ready_num_runs = sum(1 for row in expected_run_matrix if row["is_ready"] is True)
    expected_num_runs = len(expected_run_matrix)
    missing_artifacts_summary = summarize_missing_artifacts(seed_statuses, expected_run_matrix)
    if expected_num_runs == 0:
        errors.append("expected run matrix is empty")
    if ready_num_runs != expected_num_runs:
        errors.append(f"not all expected server_full runs are ready: ready={ready_num_runs}, expected={expected_num_runs}")

    created_at = utc_now_iso()
    report_dir = unique_dir(output_root / f"{dataset}_{backbone}")
    csv_rows = [csv_row(row) for row in expected_run_matrix]
    matrix_csv_path = safe_write_csv(report_dir / "server_full_expected_run_matrix.csv", csv_rows, MATRIX_FIELDS)
    is_ready_for_server_full = bool(expected_num_runs > 0 and ready_num_runs == expected_num_runs and not errors)
    report = {
        "is_valid": not errors,
        "is_ready_for_server_full": is_ready_for_server_full,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "recommendations": recommendations,
        "dataset": dataset,
        "backbone": backbone,
        "seeds": seeds,
        "shots": shots,
        "methods": checked_methods,
        "rs_cpc_prototype_inits": prototype_inits,
        "rs_cpc_M_values": rs_cpc_m_values,
        "excluded_rs_cpc_prototype_inits": excluded_inits,
        "excluded_rs_cpc_combinations": legal_rs_cpc["excluded"],
        "expected_num_runs": expected_num_runs,
        "ready_num_runs": ready_num_runs,
        "missing_artifacts_summary": missing_artifacts_summary,
        "seed_artifact_summary": seed_statuses,
        "expected_run_matrix": expected_run_matrix,
        "expected_run_matrix_csv_path": str(matrix_csv_path),
        "paper_result_safety": {
            "server_full_auto_paper_result": False,
            "requires_allow_paper_result_flag": True,
            "local_validation_is_paper_result": False,
        },
        "execution_env": execution_env,
        "run_mode": run_mode,
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
        "source_script": "scripts/check_server_full_protocol_preflight.py",
    }
    report_path = safe_write_json(report_dir / "server_full_protocol_preflight_report.json", report)
    return report_path, bool(report["is_valid"])


def inspect_seed_artifacts(
    *,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    methods: list[str],
    manifest_template: str,
    text_cache_template: str,
    adapter_plan_template: str,
    adapter_preflight_template: str,
    prototype_preflight_template: str,
) -> dict[str, Any]:
    seed_errors: list[str] = []
    warnings: list[str] = []
    context = {"dataset": dataset, "backbone": backbone, "seed": seed, "seed_token": f"seed{seed}"}

    manifest_path = resolve_single_template_path(manifest_template, context)
    manifest_entries: list[dict[str, Any]] = []
    if manifest_path is None:
        seed_errors.append("missing_manifest")
    else:
        manifest_entries, manifest_errors = read_manifest_entries(manifest_path)
        seed_errors.extend(manifest_errors)

    base_cache_paths = {
        split: find_cache_path(manifest_entries, split_ids=base_split_ids(seed), section=split)
        for split in ["val", "test"]
    }
    for split, path in base_cache_paths.items():
        if path is None:
            seed_errors.append(f"missing_base_{split}_cache")

    support_cache_paths: dict[int, str | None] = {}
    for shot in shots:
        cache_path = find_cache_path(manifest_entries, split_ids=[f"shot_{shot}_seed{seed}"], section="support")
        support_cache_paths[shot] = str(cache_path) if cache_path is not None else None
        if cache_path is None:
            seed_errors.append(f"missing_support_cache_shot_{shot}")

    text_candidates = resolve_template_paths(text_cache_template, context)
    text_status = inspect_text_cache_candidates(
        candidates=text_candidates,
        dataset=dataset,
        backbone=backbone,
        seed=seed,
    )
    if not text_status["is_ready"]:
        seed_errors.append(text_status["blocking_reason"])

    needs_adapter_artifacts = any(method in methods for method in ["tip_adapter", "proto_adapter", "rs_cpc"])
    needs_prototype_artifacts = "rs_cpc" in methods

    if needs_adapter_artifacts:
        adapter_preflight_path = resolve_single_template_path(adapter_preflight_template, context)
        adapter_preflight_status = inspect_json_artifact(
            adapter_preflight_path,
            dataset=dataset,
            backbone=backbone,
            ready_field="is_valid",
            missing_reason="adapter_input_preflight_missing",
            not_ready_reason="adapter_input_preflight_not_ready",
        )
        if not adapter_preflight_status["is_ready"]:
            seed_errors.append(adapter_preflight_status["blocking_reason"])

        adapter_plan_path = resolve_single_template_path(adapter_plan_template, context)
        adapter_plan_status = inspect_json_artifact(
            adapter_plan_path,
            dataset=dataset,
            backbone=backbone,
            ready_field="source_preflight_is_valid",
            missing_reason="adapter_input_plan_missing",
            not_ready_reason="adapter_input_plan_not_ready",
            allow_missing_ready_field=True,
        )
        if not adapter_plan_status["is_ready"]:
            seed_errors.append(adapter_plan_status["blocking_reason"])
    else:
        adapter_preflight_status = ready_optional_artifact()
        adapter_plan_status = ready_optional_artifact()

    if needs_prototype_artifacts:
        prototype_preflight_path = resolve_single_template_path(prototype_preflight_template, context)
        prototype_preflight_status = inspect_json_artifact(
            prototype_preflight_path,
            dataset=dataset,
            backbone=backbone,
            ready_field="is_valid",
            missing_reason="rs_cpc_prototype_preflight_missing",
            not_ready_reason="rs_cpc_prototype_preflight_not_ready",
        )
        if not prototype_preflight_status["is_ready"]:
            seed_errors.append(prototype_preflight_status["blocking_reason"])
    else:
        prototype_preflight_status = ready_optional_artifact()

    return {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_exists": manifest_path is not None,
        "base_cache_paths": {split: str(path) if path is not None else None for split, path in base_cache_paths.items()},
        "support_cache_paths": support_cache_paths,
        "text_cache_status": text_status,
        "adapter_preflight_status": adapter_preflight_status,
        "adapter_plan_status": adapter_plan_status,
        "prototype_preflight_status": prototype_preflight_status,
        "seed_errors": sorted(set(seed_errors)),
        "warnings": warnings,
    }


def build_seed_run_rows(
    *,
    seed: int,
    shots: list[int],
    methods: list[str],
    legal_rs_cpc: dict[str, Any],
    status: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "zero_shot" in methods:
        rows.append(make_row(method="zero_shot", seed=seed, shot=None, m_value=None, prototype_init="", status=status))
    for method in ["tip_adapter", "proto_adapter"]:
        if method not in methods:
            continue
        for shot in shots:
            rows.append(make_row(method=method, seed=seed, shot=shot, m_value=None, prototype_init="", status=status))
    if "rs_cpc" in methods:
        for combo in legal_rs_cpc["included"]:
            rows.append(
                make_row(
                    method="rs_cpc",
                    seed=seed,
                    shot=combo["shot"],
                    m_value=combo["M"],
                    prototype_init=combo["prototype_init"],
                    status=status,
                )
            )
    return rows


def make_row(
    *,
    method: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    required_inputs = ["manifest", "base_val_cache", "base_test_cache", "standalone_text_cache"]
    blocking = common_blocking_reasons(status)
    if method in {"tip_adapter", "proto_adapter", "rs_cpc"}:
        required_inputs.extend(["support_cache", "adapter_input_preflight", "adapter_input_plan"])
        if shot is not None and status["support_cache_paths"].get(shot) is None:
            blocking.append(f"missing_support_cache_shot_{shot}")
        if not status["adapter_preflight_status"]["is_ready"]:
            blocking.append(status["adapter_preflight_status"]["blocking_reason"])
        if not status["adapter_plan_status"]["is_ready"]:
            blocking.append(status["adapter_plan_status"]["blocking_reason"])
        elif not adapter_plan_row_ready(status["adapter_plan_status"].get("payload", {}), method, shot, m_value):
            blocking.append("adapter_input_plan_row_not_ready")
    if method == "rs_cpc":
        required_inputs.append("rs_cpc_prototype_preflight")
        if not status["prototype_preflight_status"]["is_ready"]:
            blocking.append(status["prototype_preflight_status"]["blocking_reason"])
        elif not prototype_combo_ready(status["prototype_preflight_status"].get("payload", {}), shot, m_value, prototype_init):
            blocking.append("rs_cpc_prototype_combo_not_ready")
    blocking = sorted(set(reason for reason in blocking if reason))
    return {
        "method": method,
        "seed": seed,
        "shot": shot,
        "M": m_value,
        "prototype_init": prototype_init,
        "required_inputs": required_inputs,
        "is_ready": not blocking,
        "blocking_reasons": blocking,
    }


def common_blocking_reasons(status: dict[str, Any]) -> list[str]:
    blocking: list[str] = []
    if not status["manifest_exists"]:
        blocking.append("missing_manifest")
    for split in ["val", "test"]:
        if status["base_cache_paths"].get(split) is None:
            blocking.append(f"missing_base_{split}_cache")
    if not status["text_cache_status"]["is_ready"]:
        blocking.append(status["text_cache_status"]["blocking_reason"])
    return blocking


def adapter_plan_row_ready(plan: dict[str, Any], method: str, shot: int | None, m_value: int | None) -> bool:
    rows = plan.get("rows")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict) or row.get("method") != method:
            continue
        if method == "zero_shot":
            return True
        if int_or_none(row.get("shot")) != shot:
            continue
        if method == "rs_cpc" and int_or_none(row.get("candidate_M")) != m_value:
            continue
        if row.get("is_ready") is True:
            return True
    return False


def prototype_combo_ready(report: dict[str, Any], shot: int | None, m_value: int | None, prototype_init: str) -> bool:
    rows = report.get("per_combination_summary")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int_or_none(row.get("shot")) != shot:
            continue
        if int_or_none(row.get("candidate_M")) != m_value:
            continue
        if row.get("prototype_init") != prototype_init:
            continue
        if row.get("is_ready") is True:
            return True
    return False


def read_manifest_entries(manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return [], [f"manifest_unreadable: {exc}"]
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return [], ["manifest_missing_entries"]
    resolved_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest_entry_not_mapping")
            continue
        merged = dict(entry)
        summary_path = resolve_path(merged.get("summary_path"), manifest_path.parent)
        if summary_path is not None and summary_path.exists():
            try:
                summary = read_json(summary_path)
                merged.update(summary)
            except Exception as exc:
                errors.append(f"feature_summary_unreadable: {summary_path}: {exc}")
        cache_path = resolve_path(merged.get("feature_cache_path") or merged.get("cache_path"), manifest_path.parent)
        if cache_path is not None:
            merged["feature_cache_path"] = str(cache_path)
        resolved_entries.append(merged)
    return resolved_entries, sorted(set(errors))


def find_cache_path(entries: list[dict[str, Any]], *, split_ids: list[str], section: str) -> Path | None:
    matches = []
    for entry in entries:
        if entry.get("split_section") != section:
            continue
        if not entry_matches_split(entry, split_ids):
            continue
        cache_path = Path(str(entry.get("feature_cache_path", "")))
        if cache_path.exists():
            matches.append(cache_path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: str(path))[-1]


def entry_matches_split(entry: dict[str, Any], split_ids: list[str]) -> bool:
    tokens = set(split_ids)
    values = [
        entry.get("split_id"),
        entry.get("split_name"),
        entry.get("split_path"),
        entry.get("summary_path"),
        entry.get("run_dir"),
        entry.get("feature_cache_path"),
    ]
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        value_tokens = {value, path.name, path.stem, *path.parts}
        if tokens & value_tokens:
            return True
    return False


def inspect_text_cache_candidates(*, candidates: list[Path], dataset: str, backbone: str, seed: int) -> dict[str, Any]:
    if not candidates:
        return {
            "is_ready": False,
            "blocking_reason": "missing_text_cache",
            "selected_text_cache_path": None,
            "candidates": [],
        }
    inspections = [inspect_text_cache(path, dataset=dataset, backbone=backbone, seed=seed) for path in candidates]
    ready = [inspection for inspection in inspections if inspection["is_ready"]]
    if ready:
        selected = sorted(ready, key=lambda item: (str(item.get("created_at") or ""), str(item["path"])))[-1]
        return {
            "is_ready": True,
            "blocking_reason": "",
            "selected_text_cache_path": selected["path"],
            "candidates": inspections,
        }
    blocking = "text_cache_dry_run_or_fake" if any(item.get("dry_run") or item.get("uses_fake_text_features") for item in inspections) else "text_cache_not_ready"
    return {
        "is_ready": False,
        "blocking_reason": blocking,
        "selected_text_cache_path": None,
        "candidates": inspections,
    }


def inspect_text_cache(path: Path, *, dataset: str, backbone: str, seed: int) -> dict[str, Any]:
    inspection: dict[str, Any] = {"path": str(path), "exists": path.exists(), "is_ready": False}
    if not path.exists():
        inspection["blocking_reason"] = "missing_text_cache"
        return inspection
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        inspection["blocking_reason"] = "text_cache_unreadable"
        inspection["error"] = str(exc)
        return inspection
    if not isinstance(data, dict):
        inspection["blocking_reason"] = "text_cache_not_mapping"
        return inspection
    dry_run = bool(data.get("dry_run", False))
    fake = bool(data.get("uses_fake_text_features", False))
    base_split = str(data.get("base_split", ""))
    inspection.update(
        {
            "dataset": data.get("dataset"),
            "backbone": data.get("backbone"),
            "base_split": base_split,
            "dry_run": dry_run,
            "uses_fake_text_features": fake,
            "is_paper_result": data.get("is_paper_result"),
            "created_at": data.get("created_at"),
        }
    )
    errors = []
    if data.get("dataset") != dataset:
        errors.append("text_cache_dataset_mismatch")
    if data.get("backbone") != backbone:
        errors.append("text_cache_backbone_mismatch")
    if base_split and base_split not in set(base_split_ids(seed)):
        errors.append("text_cache_base_split_mismatch")
    if dry_run or fake:
        errors.append("text_cache_dry_run_or_fake")
    if data.get("is_paper_result") is not False:
        errors.append("text_cache_is_paper_result_not_false")
    inspection["blocking_reasons"] = errors
    inspection["is_ready"] = not errors
    return inspection


def inspect_json_artifact(
    path: Path | None,
    *,
    dataset: str,
    backbone: str,
    ready_field: str,
    missing_reason: str,
    not_ready_reason: str,
    allow_missing_ready_field: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "is_ready": False, "blocking_reason": missing_reason, "payload": {}}
    try:
        payload = read_json(path)
    except Exception as exc:
        return {
            "path": str(path),
            "exists": path.exists(),
            "is_ready": False,
            "blocking_reason": not_ready_reason,
            "error": str(exc),
            "payload": {},
        }
    reasons = []
    if payload.get("dataset") != dataset:
        reasons.append(f"{not_ready_reason}_dataset_mismatch")
    if payload.get("backbone") != backbone:
        reasons.append(f"{not_ready_reason}_backbone_mismatch")
    if ready_field in payload:
        if payload.get(ready_field) is not True:
            reasons.append(not_ready_reason)
    elif not allow_missing_ready_field:
        reasons.append(not_ready_reason)
    elif payload.get("is_valid") is False:
        reasons.append(not_ready_reason)
    return {
        "path": str(path),
        "exists": True,
        "is_ready": not reasons,
        "blocking_reason": reasons[0] if reasons else "",
        "blocking_reasons": reasons,
        "payload": payload,
    }


def ready_optional_artifact() -> dict[str, Any]:
    return {"path": None, "exists": False, "is_ready": True, "blocking_reason": "", "payload": {}}


def legal_rs_cpc_combinations(
    *,
    shots: list[int],
    prototype_inits: list[str],
    m_values: list[int],
) -> dict[str, list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for shot in shots:
        for init_mode in prototype_inits:
            for m_value in m_values:
                reason = rs_cpc_exclusion_reason(shot=shot, init_mode=init_mode, m_value=m_value)
                row = {"shot": shot, "M": m_value, "prototype_init": init_mode}
                if reason:
                    excluded.append({**row, "reason": reason})
                else:
                    included.append(row)
    return {"included": included, "excluded": excluded}


def rs_cpc_exclusion_reason(*, shot: int, init_mode: str, m_value: int) -> str:
    if init_mode not in SUPPORTED_RS_CPC_INITS:
        return "unsupported_or_excluded_prototype_init"
    if init_mode == "mean" and m_value != 1:
        return "mean_only_supports_M_1"
    if m_value > shot:
        return "M_exceeds_shot"
    return ""


def normalize_methods(methods: list[str], errors: list[str]) -> list[str]:
    normalized: list[str] = []
    for method in methods:
        if method not in SUPPORTED_METHODS:
            errors.append(f"unsupported method requested: {method}")
            continue
        if method not in normalized:
            normalized.append(method)
    return normalized


def normalize_rs_cpc_inits(values: list[str]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    excluded: list[str] = []
    for value in values:
        init_mode = str(value)
        if init_mode not in SUPPORTED_RS_CPC_INITS:
            if init_mode not in excluded:
                excluded.append(init_mode)
            continue
        if init_mode not in normalized:
            normalized.append(init_mode)
    return normalized, excluded


def summarize_missing_artifacts(seed_statuses: dict[int, dict[str, Any]], matrix: list[dict[str, Any]]) -> dict[str, Any]:
    seed_error_counts = Counter()
    for status in seed_statuses.values():
        seed_error_counts.update(status.get("seed_errors", []))
    row_blocking_counts = Counter()
    for row in matrix:
        row_blocking_counts.update(row.get("blocking_reasons", []))
    return {
        "seed_error_counts": dict(sorted(seed_error_counts.items())),
        "row_blocking_counts": dict(sorted(row_blocking_counts.items())),
    }


def resolve_template_paths(template: str, context: dict[str, Any]) -> list[Path]:
    rendered = template.format(**context)
    if any(character in rendered for character in ["*", "?", "["]):
        return [Path(path) for path in sorted(glob.glob(rendered))]
    path = Path(rendered)
    return [path] if path.exists() else []


def resolve_single_template_path(template: str, context: dict[str, Any]) -> Path | None:
    paths = resolve_template_paths(template, context)
    if not paths:
        return None
    return sorted(paths, key=lambda path: str(path))[-1]


def resolve_path(value: Any, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    return candidate if candidate.exists() else path


def base_split_ids(seed: int) -> list[str]:
    return [f"base_seed{seed}", f"base_split_seed{seed}"]


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


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "seed": row["seed"],
        "shot": "" if row["shot"] is None else row["shot"],
        "M": "" if row["M"] is None else row["M"],
        "prototype_init": row["prototype_init"],
        "required_inputs": ";".join(row["required_inputs"]),
        "is_ready": row["is_ready"],
        "blocking_reasons": ";".join(row["blocking_reasons"]),
    }


def ensure_not_results_raw(output_dir: Path) -> None:
    parts = output_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "results" and parts[index + 1] == "raw":
            raise ValueError("server_full protocol preflight reports must not be written under results/raw")


def unique_dir(base_dir: Path) -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique server_full protocol preflight directory under {base_dir}")


if __name__ == "__main__":
    main()
