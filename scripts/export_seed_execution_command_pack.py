#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging.system_info import git_commit_hash
from src.utils.io import read_json, safe_write_json
from src.utils.timing import utc_now_iso


PHASES = [
    ("A", "dataset/split readiness"),
    ("B", "image feature extraction and manifest"),
    ("C", "text feature preflight + text feature extraction"),
    ("D", "adapter input preflight"),
    ("E", "adapter input plan"),
    ("F", "RS-CPC prototype preflight"),
    ("G", "rerun server_full protocol preflight"),
]
SCRIPT_PATHS = {
    "generate_splits": "scripts/generate_splits.py",
    "extract_features": "scripts/extract_features.py",
    "build_feature_cache_manifest": "scripts/build_feature_cache_manifest.py",
    "check_text_feature_cache_preflight": "scripts/check_text_feature_cache_preflight.py",
    "extract_text_features": "scripts/extract_text_features.py",
    "check_adapter_input_preflight": "scripts/check_adapter_input_preflight.py",
    "export_adapter_input_plan": "scripts/export_adapter_input_plan.py",
    "check_rs_cpc_prototype_preflight": "scripts/check_rs_cpc_prototype_preflight.py",
    "check_server_full_protocol_preflight": "scripts/check_server_full_protocol_preflight.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a review-only seed execution command pack.")
    parser.add_argument("--seed-expansion-plan", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/seed_execution_command_packs")
    parser.add_argument("--execution-env", required=True)
    parser.add_argument("--run-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_seed_execution_command_pack(
        seed_expansion_plan=args.seed_expansion_plan,
        dataset=args.dataset,
        backbone=args.backbone,
        seed=args.seed,
        shots=args.shots,
        output_dir=args.output_dir,
        execution_env=args.execution_env,
        run_mode=args.run_mode,
        command=shlex.join(sys.argv),
    )
    print(f"seed_execution_command_pack_dir={result['pack_dir']}")
    print(f"seed_execution_command_pack_md={result['markdown_path']}")
    print(f"seed_execution_command_pack_sh={result['shell_path']}")
    print(f"seed_execution_command_pack_json={result['json_path']}")
    print(f"num_command_items={result['num_command_items']}")


def export_seed_execution_command_pack(
    *,
    seed_expansion_plan: str | Path,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    output_dir: str | Path,
    execution_env: str,
    run_mode: str,
    command: str | None = None,
) -> dict[str, Any]:
    plan_path = Path(seed_expansion_plan)
    output_root = Path(output_dir)
    ensure_not_results_raw(output_root)
    plan_bytes = plan_path.read_bytes()
    plan = read_json(plan_path)
    validate_plan_context(plan, dataset=dataset, backbone=backbone, seed=seed)

    script_inventory = inspect_required_scripts()
    plan_items = [item for item in plan.get("plan_items", []) if isinstance(item, dict) and item.get("seed") == seed]
    command_items = build_command_items(dataset=dataset, backbone=backbone, seed=seed, shots=shots, plan_items=plan_items)
    created_at = utc_now_iso()
    pack_dir = unique_pack_dir(output_root, dataset=dataset, backbone=backbone, seed=seed)
    basename = f"seed{seed}_command_pack"

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
        "source_script": "scripts/export_seed_execution_command_pack.py",
        "source_seed_expansion_plan": str(plan_path),
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "shots": shots,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "review_only": True,
        "do_not_run_blindly": True,
        "script_inventory": script_inventory,
        "num_command_items": len(command_items),
        "command_items": command_items,
    }
    json_path = safe_write_json(pack_dir / f"{basename}.json", json_payload)
    markdown_path = write_text_no_overwrite(
        pack_dir / f"{basename}.md",
        render_markdown_pack(
            dataset=dataset,
            backbone=backbone,
            seed=seed,
            shots=shots,
            source_plan=plan_path,
            created_at=created_at,
            script_inventory=script_inventory,
            command_items=command_items,
        ),
    )
    shell_path = write_text_no_overwrite(
        pack_dir / f"{basename}.sh",
        render_shell_pack(dataset=dataset, backbone=backbone, seed=seed, command_items=command_items),
    )

    if plan_path.exists() and plan_path.read_bytes() != plan_bytes:
        raise RuntimeError(f"input seed expansion plan was modified unexpectedly: {plan_path}")
    return {
        "pack_dir": str(pack_dir),
        "markdown_path": str(markdown_path),
        "shell_path": str(shell_path),
        "json_path": str(json_path),
        "num_command_items": len(command_items),
        "command_items": command_items,
    }


def build_command_items(
    *,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    plan_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan_by_phase = {str(item.get("phase")): item for item in plan_items}
    return [
        command_item(
            phase="A",
            title="Dataset and split readiness",
            step_kind="local_validation/preflight",
            scripts=["scripts/generate_splits.py"],
            commands=[
                split_command(dataset=dataset, seed=seed, shots=shots),
            ],
            plan_item=plan_by_phase.get("A"),
        ),
        command_item(
            phase="B",
            title="Image feature extraction and manifest",
            step_kind="feature extraction artifact generation",
            scripts=["scripts/extract_features.py", "scripts/build_feature_cache_manifest.py"],
            commands=[
                image_feature_command(dataset=dataset, backbone=backbone, seed=seed),
                manifest_command(backbone=backbone, dataset=dataset, seed=seed),
            ],
            plan_item=plan_by_phase.get("B"),
        ),
        command_item(
            phase="C",
            title="Text feature preflight and extraction",
            step_kind="feature extraction artifact generation",
            scripts=["scripts/check_text_feature_cache_preflight.py", "scripts/extract_text_features.py"],
            commands=[
                text_preflight_command(dataset=dataset, backbone=backbone, seed=seed),
                text_feature_command(dataset=dataset, backbone=backbone, seed=seed),
            ],
            plan_item=plan_by_phase.get("C"),
        ),
        command_item(
            phase="D",
            title="Adapter input preflight",
            step_kind="local_validation/preflight",
            scripts=["scripts/check_adapter_input_preflight.py"],
            commands=[
                adapter_preflight_command(dataset=dataset, backbone=backbone, seed=seed, shots=shots),
            ],
            plan_item=plan_by_phase.get("D"),
        ),
        command_item(
            phase="E",
            title="Adapter input plan",
            step_kind="local_validation/preflight",
            scripts=["scripts/export_adapter_input_plan.py"],
            commands=[
                adapter_plan_command(dataset=dataset, backbone=backbone, seed=seed),
            ],
            plan_item=plan_by_phase.get("E"),
        ),
        command_item(
            phase="F",
            title="RS-CPC prototype preflight",
            step_kind="local_validation/preflight",
            scripts=["scripts/check_rs_cpc_prototype_preflight.py"],
            commands=[
                prototype_preflight_command(dataset=dataset, backbone=backbone, seed=seed),
            ],
            plan_item=plan_by_phase.get("F"),
        ),
        command_item(
            phase="G",
            title="Rerun server_full protocol preflight",
            step_kind="server_full protocol preflight",
            scripts=["scripts/check_server_full_protocol_preflight.py"],
            commands=[
                server_full_preflight_command(dataset=dataset, backbone=backbone, shots=shots),
            ],
            plan_item=plan_by_phase.get("G"),
        ),
    ]


def command_item(
    *,
    phase: str,
    title: str,
    step_kind: str,
    scripts: list[str],
    commands: list[str],
    plan_item: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "title": title,
        "step_kind": step_kind,
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
        "source_plan_status": (plan_item or {}).get("current_status", "unknown"),
        "source_plan_blocking_reasons": (plan_item or {}).get("blocking_reason_from_report", []),
        "scripts": scripts,
        "commands": commands,
        "review_note": "Verify exact paths, server environment, and TODO placeholders before manual execution.",
    }


def split_command(*, dataset: str, seed: int, shots: list[int]) -> str:
    return "\n".join(
        [
            "python3 scripts/generate_splits.py \\",
            f"  --config configs/datasets/{dataset}.yaml \\",
            f"  --dataset {dataset} \\",
            "  --dataset-root <DATASET_ROOT_TODO> \\",
            f"  --shots {' '.join(str(shot) for shot in shots)} \\",
            f"  --seeds {seed} \\",
            f"  --output-dir splits/{dataset}",
        ]
    )


def image_feature_command(*, dataset: str, backbone: str, seed: int) -> str:
    return "\n".join(
        [
            "# TODO verify exact feature extraction workflow before running.",
            "python3 scripts/extract_features.py \\",
            f"  --dataset {dataset} \\",
            f"  --backbone {backbone} \\",
            f"  --config configs/backbones/{backbone}.yaml \\",
            f"  --split splits/{dataset}/base_split_seed{seed}.json \\",
            "  --weights-path <WEIGHTS_PATH_TODO> \\",
            "  --device cuda \\",
            "  --execution-env remote_server \\",
            "  --run-mode server_full \\",
            f"  --output-dir outputs/features/{backbone}/{dataset}/base_seed{seed}",
        ]
    )


def manifest_command(*, backbone: str, dataset: str, seed: int) -> str:
    return "\n".join(
        [
            "python3 scripts/build_feature_cache_manifest.py \\",
            f"  --feature-root outputs/features/{backbone}/{dataset}/base_seed{seed} \\",
            f"  --output-dir outputs/manifests/feature_cache_after_seed{seed}_support",
        ]
    )


def text_preflight_command(*, dataset: str, backbone: str, seed: int) -> str:
    return "\n".join(
        [
            "python3 scripts/check_text_feature_cache_preflight.py \\",
            f"  --manifest outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json \\",
            f"  --dataset {dataset} \\",
            f"  --backbone {backbone} \\",
            f"  --base-split base_seed{seed} \\",
            "  --output-dir outputs/preflight/text_features \\",
            "  --execution-env remote_server \\",
            "  --run-mode local_validation",
        ]
    )


def text_feature_command(*, dataset: str, backbone: str, seed: int) -> str:
    return "\n".join(
        [
            "python3 scripts/extract_text_features.py \\",
            f"  --dataset {dataset} \\",
            f"  --backbone {backbone} \\",
            f"  --base-split base_seed{seed} \\",
            "  --preflight-report <TEXT_FEATURE_CACHE_PREFLIGHT_REPORT_TODO> \\",
            f"  --backbone-config configs/backbones/{backbone}.yaml \\",
            "  --method-config configs/methods/zero_shot_clip.yaml \\",
            "  --weights-path <WEIGHTS_PATH_TODO> \\",
            "  --output-dir outputs/features \\",
            "  --device cuda \\",
            "  --execution-env remote_server \\",
            "  --run-mode server_full",
        ]
    )


def adapter_preflight_command(*, dataset: str, backbone: str, seed: int, shots: list[int]) -> str:
    shot_splits = " ".join(f"shot_{shot}_seed{seed}" for shot in shots)
    return "\n".join(
        [
            "python3 scripts/check_adapter_input_preflight.py \\",
            f"  --manifest outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json \\",
            f"  --dataset {dataset} \\",
            f"  --backbone {backbone} \\",
            f"  --base-split base_seed{seed} \\",
            f"  --shot-splits {shot_splits} \\",
            "  --methods tip_adapter proto_adapter rs_cpc \\",
            "  --output-dir outputs/preflight/adapter_input \\",
            "  --execution-env remote_server \\",
            "  --run-mode local_validation",
        ]
    )


def adapter_plan_command(*, dataset: str, backbone: str, seed: int) -> str:
    return "\n".join(
        [
            "python3 scripts/export_adapter_input_plan.py \\",
            f"  --preflight-report outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json \\",
            "  --output-dir outputs/preflight/adapter_input_plans \\",
            "  --execution-env remote_server \\",
            "  --run-mode local_validation",
        ]
    )


def prototype_preflight_command(*, dataset: str, backbone: str, seed: int) -> str:
    return "\n".join(
        [
            f"# Expected output pattern: outputs/preflight/rs_cpc_prototypes/{dataset}_{backbone}_seed{seed}/<TIMESTAMP_TODO>/rs_cpc_prototype_preflight_report.json",
            "python3 scripts/check_rs_cpc_prototype_preflight.py \\",
            f"  --adapter-input-plan outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/<ADAPTER_INPUT_PLAN_TIMESTAMP_TODO>/adapter_input_plan.json \\",
            f"  --preflight-report outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json \\",
            "  --prototype-inits mean random_group_mean medoid \\",
            "  --output-dir outputs/preflight/rs_cpc_prototypes \\",
            "  --execution-env remote_server \\",
            "  --run-mode local_validation",
        ]
    )


def server_full_preflight_command(*, dataset: str, backbone: str, shots: list[int]) -> str:
    return "\n".join(
        [
            "python3 scripts/check_server_full_protocol_preflight.py \\",
            f"  --dataset {dataset} \\",
            f"  --backbone {backbone} \\",
            "  --seeds <COMPLETE_SERVER_FULL_SEED_LIST_TODO> \\",
            f"  --shots {' '.join(str(shot) for shot in shots)} \\",
            "  --methods zero_shot tip_adapter proto_adapter rs_cpc \\",
            "  --rs-cpc-prototype-inits mean random_group_mean medoid \\",
            "  --rs-cpc-M-values 1 2 4 8 \\",
            "  --manifest-template 'outputs/manifests/feature_cache_after_seed{seed}_support/feature_cache_manifest.json' \\",
            "  --text-cache-template 'outputs/features/{backbone}/{dataset}/base_seed{seed}/{dataset}/{backbone}/text/*/text_feature_cache.pt' \\",
            "  --adapter-plan-template 'outputs/preflight/adapter_input_plans/{dataset}_{backbone}_seed{seed}/*/adapter_input_plan.json' \\",
            "  --adapter-preflight-template 'outputs/preflight/adapter_input/{dataset}_{backbone}_seed{seed}/adapter_input_preflight_report.json' \\",
            "  --prototype-preflight-template 'outputs/preflight/rs_cpc_prototypes/{dataset}_{backbone}_seed{seed}/*/rs_cpc_prototype_preflight_report.json' \\",
            "  --output-dir outputs/preflight/server_full_protocol \\",
            "  --execution-env remote_server \\",
            "  --run-mode local_validation",
        ]
    )


def inspect_required_scripts() -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name, script_path in SCRIPT_PATHS.items():
        path = Path(script_path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        inventory[name] = {
            "path": script_path,
            "exists": path.exists(),
            "has_argparse": "argparse" in text,
            "has_parse_args": "parse_args" in text,
        }
    return inventory


def validate_plan_context(plan: dict[str, Any], *, dataset: str, backbone: str, seed: int) -> None:
    if plan.get("dataset") != dataset:
        raise ValueError(f"seed expansion plan dataset mismatch, expected {dataset}, found {plan.get('dataset')}")
    if plan.get("backbone") != backbone:
        raise ValueError(f"seed expansion plan backbone mismatch, expected {backbone}, found {plan.get('backbone')}")
    target_seeds = plan.get("target_seeds")
    if isinstance(target_seeds, list) and seed not in target_seeds:
        raise ValueError(f"seed {seed} is not listed in seed expansion plan target_seeds={target_seeds}")


def render_markdown_pack(
    *,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    source_plan: Path,
    created_at: str,
    script_inventory: dict[str, Any],
    command_items: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Seed {seed} Execution Command Pack",
        "",
        "This command pack is generated for review only.",
        "It does not run experiments.",
        "It is not a paper result.",
        "Do not run blindly. Verify every TODO placeholder and server path first.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Backbone: `{backbone}`",
        f"- Seed: `{seed}`",
        f"- Shots: `{', '.join(str(shot) for shot in shots)}`",
        f"- Source seed expansion plan: `{source_plan}`",
        f"- Created at: `{created_at}`",
        "",
        "Script availability:",
        "",
    ]
    for name, info in script_inventory.items():
        lines.append(f"- `{info['path']}`: exists={str(info['exists']).lower()}, argparse={str(info['has_argparse']).lower()}")
    lines.append("")
    for item in command_items:
        lines.extend(
            [
                f"## Phase {item['phase']}: {item['title']}",
                "",
                f"- Step type: `{item['step_kind']}`",
                "- Paper result: `false`",
                "- Writes `results/raw`: `false`",
                f"- Source plan status: `{item['source_plan_status']}`",
                f"- Source blocking reasons: `{', '.join(str(reason) for reason in item['source_plan_blocking_reasons'])}`",
                "",
            ]
        )
        for command in item["commands"]:
            lines.extend(["```bash", command, "```", ""])
    return "\n".join(lines)


def render_shell_pack(*, dataset: str, backbone: str, seed: int, command_items: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'echo "This command pack is generated for review. Do not run blindly."',
        "exit 1",
        "",
        ": <<'COMMAND_PACK'",
        f"# Review-only command pack for {dataset}/{backbone}/seed{seed}",
        "# It is not a paper result and must not be executed without manual review.",
        "",
    ]
    for item in command_items:
        lines.extend([f"# Phase {item['phase']}: {item['title']}", f"# Step type: {item['step_kind']}"])
        for command in item["commands"]:
            lines.extend([command, ""])
    lines.append("COMMAND_PACK")
    lines.append("")
    return "\n".join(lines)


def unique_pack_dir(output_dir: Path, *, dataset: str, backbone: str, seed: int) -> Path:
    base_dir = output_dir / f"{dataset}_{backbone}_seed{seed}"
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    for index in range(1000):
        candidate = base_dir / (stamp if index == 0 else f"{stamp}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique seed execution command pack directory under {base_dir}")


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
            raise ValueError("seed execution command packs must not be written under results/raw")


if __name__ == "__main__":
    main()
