from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export_server_full_evaluation_command_pack import export_server_full_evaluation_command_pack
from src.utils.io import read_json, safe_write_json


class ServerFullEvaluationCommandPackTest(unittest.TestCase):
    def test_ready_report_generates_132_command_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            payload = read_json(result["json_path"])
            self.assertEqual(result["num_commands"], 132)
            self.assertEqual(payload["num_commands"], 132)
            self.assertEqual(len(payload["command_rows"]), 132)
            self.assertTrue(Path(result["markdown_path"]).exists())
            self.assertTrue(Path(result["shell_path"]).exists())
            self.assertTrue(Path(result["matrix_csv_path"]).exists())
            self.assertFalse((root / "results" / "raw").exists())

    def test_not_ready_report_refuses_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=False)

            with self.assertRaisesRegex(ValueError, "not ready|not valid|fully ready"):
                export_pack(root, report_path)

    def test_shell_pack_exits_before_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            shell_text = Path(result["shell_path"]).read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", shell_text)
            self.assertIn('echo "This command pack is generated for review. Do not run blindly."', shell_text)
            self.assertLess(shell_text.index("exit 1"), shell_text.index("python3 scripts/run_zero_shot.py"))
            self.assertIn(": <<'SERVER_FULL_COMMAND_PACK'", shell_text)

    def test_commands_use_seed_specific_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            payload = read_json(result["json_path"])
            zero_seed2 = next(row for row in payload["command_rows"] if row["method"] == "zero_shot" and row["seed"] == 2)
            self.assertIn("feature_cache_after_seed2_support/feature_cache_manifest.json", zero_seed2["manifest_path"])
            self.assertIn("base_seed2/eurosat/remoteclip_vit_b32/text", zero_seed2["text_feature_cache_path"])
            self.assertIn("--skip-preflight-check", zero_seed2["command"])
            tip_seed2 = next(row for row in payload["command_rows"] if row["method"] == "tip_adapter" and row["seed"] == 2 and row["shot"] == 4)
            self.assertIn("adapter_input_plans/eurosat_remoteclip_vit_b32_seed2", tip_seed2["adapter_plan_path"])
            self.assertIn("adapter_input/eurosat_remoteclip_vit_b32_seed2", tip_seed2["adapter_preflight_path"])
            self.assertIn("--shot-split shot_4_seed2", tip_seed2["command"])
            rs_seed2 = next(
                row
                for row in payload["command_rows"]
                if row["method"] == "rs_cpc" and row["seed"] == 2 and row["shot"] == 8 and row["M"] == 4 and row["prototype_init"] == "medoid"
            )
            self.assertIn("rs_cpc_prototypes/eurosat_remoteclip_vit_b32_seed2", rs_seed2["prototype_preflight_path"])
            self.assertIn("--prototype-init medoid", rs_seed2["command"])

    def test_rs_cpc_legal_combo_count_is_correct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            rows = read_json(result["json_path"])["command_rows"]
            rs_rows = [row for row in rows if row["method"] == "rs_cpc"]
            self.assertEqual(len(rs_rows), 99)
            self.assertEqual(sum(1 for row in rs_rows if row["prototype_init"] == "mean"), 15)
            self.assertEqual(sum(1 for row in rs_rows if row["prototype_init"] == "random_group_mean"), 42)
            self.assertEqual(sum(1 for row in rs_rows if row["prototype_init"] == "medoid"), 42)
            self.assertFalse(any(row["prototype_init"] == "mean" and row["M"] != 1 for row in rs_rows))
            self.assertFalse(any(row["M"] > row["shot"] for row in rs_rows))

    def test_command_rows_require_post_run_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            rows = read_json(result["json_path"])["command_rows"]
            self.assertTrue(all(row["requires_post_run_preflight"] is True for row in rows))
            self.assertTrue(all(row["is_paper_result"] is False for row in rows))

    def test_exporter_metadata_safety_flags_are_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = write_fake_server_full_report(root, ready=True)

            result = export_pack(root, report_path)

            payload = read_json(result["json_path"])
            for flag in [
                "is_paper_result",
                "writes_results_raw",
                "computes_logits",
                "computes_accuracy",
                "evaluates_model",
                "trains_model",
                "modifies_results",
                "deletes_results",
            ]:
                self.assertFalse(payload[flag], flag)


def export_pack(root: Path, report_path: Path) -> dict[str, object]:
    return export_server_full_evaluation_command_pack(
        server_full_preflight_report=report_path,
        dataset="eurosat",
        backbone="remoteclip_vit_b32",
        seeds=[1, 2, 3],
        shots=[1, 2, 4, 8, 16],
        methods=["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"],
        output_dir=root / "outputs" / "analysis" / "server_full_command_packs",
        execution_env="remote_server",
        run_mode="server_full",
        paper_result_mode="candidate_only",
        command="unit test export server full command pack",
    )


def write_fake_server_full_report(root: Path, *, ready: bool) -> Path:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    seeds = [1, 2, 3]
    shots = [1, 2, 4, 8, 16]
    methods = ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"]
    seed_artifact_summary = {str(seed): seed_artifacts(root, dataset=dataset, backbone=backbone, seed=seed) for seed in seeds}
    expected_run_matrix = expected_rows(seeds=seeds, shots=shots, methods=methods, ready=ready)
    report = {
        "is_valid": ready,
        "is_ready_for_server_full": ready,
        "errors": [] if ready else ["not_ready"],
        "warnings": [],
        "dataset": dataset,
        "backbone": backbone,
        "seeds": seeds,
        "shots": shots,
        "methods": methods,
        "expected_num_runs": len(expected_run_matrix),
        "ready_num_runs": len(expected_run_matrix) if ready else 0,
        "seed_artifact_summary": seed_artifact_summary,
        "expected_run_matrix": expected_run_matrix,
        "execution_env": "remote_server",
        "run_mode": "local_validation",
        "is_paper_result": False,
        "writes_results_raw": False,
        "computes_logits": False,
        "computes_accuracy": False,
        "evaluates_model": False,
        "trains_model": False,
        "modifies_results": False,
        "deletes_results": False,
    }
    path = root / "outputs" / "preflight" / "server_full_protocol" / "eurosat_remoteclip_vit_b32" / "20260513T000000" / "server_full_protocol_preflight_report.json"
    return safe_write_json(path, report)


def seed_artifacts(root: Path, *, dataset: str, backbone: str, seed: int) -> dict[str, object]:
    manifest_path = root / "outputs" / "manifests" / f"feature_cache_after_seed{seed}_support" / "feature_cache_manifest.json"
    text_path = (
        root
        / "outputs"
        / "features"
        / backbone
        / dataset
        / f"base_seed{seed}"
        / dataset
        / backbone
        / "text"
        / "20260513T000000"
        / "text_feature_cache.pt"
    )
    adapter_preflight_path = root / "outputs" / "preflight" / "adapter_input" / f"{dataset}_{backbone}_seed{seed}" / "adapter_input_preflight_report.json"
    adapter_plan_path = root / "outputs" / "preflight" / "adapter_input_plans" / f"{dataset}_{backbone}_seed{seed}" / "20260513T000000" / "adapter_input_plan.json"
    prototype_preflight_path = root / "outputs" / "preflight" / "rs_cpc_prototypes" / f"{dataset}_{backbone}_seed{seed}" / "20260513T000000" / "rs_cpc_prototype_preflight_report.json"
    for path in [manifest_path, text_path, adapter_preflight_path, adapter_plan_path, prototype_preflight_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": True,
        "text_cache_status": {
            "is_ready": True,
            "selected_text_cache_path": str(text_path),
        },
        "adapter_preflight_status": {
            "is_ready": True,
            "path": str(adapter_preflight_path),
        },
        "adapter_plan_status": {
            "is_ready": True,
            "path": str(adapter_plan_path),
        },
        "prototype_preflight_status": {
            "is_ready": True,
            "path": str(prototype_preflight_path),
        },
        "seed_errors": [],
        "warnings": [],
    }


def expected_rows(*, seeds: list[int], shots: list[int], methods: list[str], ready: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        if "zero_shot" in methods:
            rows.append(row(method="zero_shot", seed=seed, shot=None, m_value=None, prototype_init="", ready=ready))
        for method in ["tip_adapter", "proto_adapter"]:
            if method not in methods:
                continue
            for shot in shots:
                rows.append(row(method=method, seed=seed, shot=shot, m_value=None, prototype_init="", ready=ready))
        if "rs_cpc" in methods:
            for shot in shots:
                for init_mode in ["mean", "random_group_mean", "medoid"]:
                    for m_value in [1, 2, 4, 8]:
                        if init_mode == "mean" and m_value != 1:
                            continue
                        if m_value > shot:
                            continue
                        rows.append(row(method="rs_cpc", seed=seed, shot=shot, m_value=m_value, prototype_init=init_mode, ready=ready))
    return rows


def row(
    *,
    method: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str,
    ready: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "shot": shot,
        "M": m_value,
        "prototype_init": prototype_init,
        "required_inputs": [],
        "is_ready": ready,
        "blocking_reasons": [] if ready else ["not_ready"],
    }


if __name__ == "__main__":
    unittest.main()
