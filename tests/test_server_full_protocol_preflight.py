from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.check_server_full_protocol_preflight import run_server_full_protocol_preflight
from src.utils.io import read_json, safe_write_json


class ServerFullProtocolPreflightTest(unittest.TestCase):
    def test_complete_seed_1_2_3_artifact_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in [1, 2, 3]:
                write_complete_seed_artifacts(root, seed=seed)

            report_path, is_valid = run_preflight(root, seeds=[1, 2, 3])

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["is_ready_for_server_full"])
            self.assertEqual(report["expected_num_runs"], 132)
            self.assertEqual(report["ready_num_runs"], 132)
            self.assertEqual(len(report["expected_run_matrix"]), 132)
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["writes_results_raw"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["modifies_results"])
            self.assertFalse(report["deletes_results"])
            self.assertTrue(Path(report["expected_run_matrix_csv_path"]).exists())
            self.assertFalse((root / "results" / "raw").exists())

    def test_missing_seed_2_and_3_artifacts_make_server_full_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_complete_seed_artifacts(root, seed=1)

            report_path, is_valid = run_preflight(root, seeds=[1, 2, 3])

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertFalse(report["is_ready_for_server_full"])
            self.assertEqual(report["expected_num_runs"], 132)
            self.assertEqual(report["ready_num_runs"], 44)
            self.assertIn("seed2: missing_manifest", report["errors"])
            self.assertIn("seed3: missing_manifest", report["errors"])
            self.assertGreater(report["missing_artifacts_summary"]["row_blocking_counts"]["missing_manifest"], 0)

    def test_illegal_rs_cpc_combos_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_complete_seed_artifacts(root, seed=1, shots=[1, 2], m_values=[1, 2], prototype_inits=["mean", "random_group_mean", "medoid"])

            report_path, is_valid = run_preflight(
                root,
                seeds=[1],
                shots=[1, 2],
                methods=["rs_cpc"],
                prototype_inits=["mean", "random_group_mean", "medoid", "kmeans"],
                m_values=[1, 2],
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            rows = report["expected_run_matrix"]
            self.assertEqual(report["expected_num_runs"], 8)
            self.assertFalse(any(row["prototype_init"] == "kmeans" for row in rows))
            self.assertFalse(any(row["prototype_init"] == "mean" and row["M"] != 1 for row in rows))
            self.assertIn("kmeans", report["excluded_rs_cpc_prototype_inits"])
            excluded_reasons = {row["reason"] for row in report["excluded_rs_cpc_combinations"]}
            self.assertIn("mean_only_supports_M_1", excluded_reasons)
            self.assertIn("M_exceeds_shot", excluded_reasons)

    def test_mean_rs_cpc_rows_are_only_m1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_complete_seed_artifacts(root, seed=1)

            report_path, is_valid = run_preflight(root, seeds=[1], methods=["rs_cpc"])

            report = read_json(report_path)
            self.assertTrue(is_valid)
            mean_rows = [row for row in report["expected_run_matrix"] if row["prototype_init"] == "mean"]
            self.assertEqual(len(mean_rows), 5)
            self.assertTrue(all(row["M"] == 1 for row in mean_rows))

    def test_dry_run_text_cache_blocks_server_full_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_complete_seed_artifacts(root, seed=1, text_dry_run=True)

            report_path, is_valid = run_preflight(root, seeds=[1], methods=["zero_shot"])

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_ready_for_server_full"])
            self.assertEqual(report["expected_num_runs"], 1)
            self.assertEqual(report["ready_num_runs"], 0)
            self.assertIn("seed1: text_cache_dry_run_or_fake", report["errors"])
            self.assertEqual(report["expected_run_matrix"][0]["blocking_reasons"], ["text_cache_dry_run_or_fake"])

    def test_expected_run_count_matches_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in [1, 2, 3]:
                write_complete_seed_artifacts(root, seed=seed)

            report_path, _ = run_preflight(root, seeds=[1, 2, 3])

            report = read_json(report_path)
            self.assertEqual(report["expected_num_runs"], len(report["expected_run_matrix"]))
            self.assertEqual(report["ready_num_runs"], sum(1 for row in report["expected_run_matrix"] if row["is_ready"]))

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_complete_seed_artifacts(root, seed=1)

            with self.assertRaisesRegex(ValueError, "results/raw"):
                run_preflight(root, seeds=[1], output_dir=root / "results" / "raw" / "bad")


def run_preflight(
    root: Path,
    *,
    seeds: list[int],
    shots: list[int] | None = None,
    methods: list[str] | None = None,
    prototype_inits: list[str] | None = None,
    m_values: list[int] | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    return run_server_full_protocol_preflight(
        dataset="eurosat",
        backbone="remoteclip_vit_b32",
        seeds=seeds,
        shots=shots or [1, 2, 4, 8, 16],
        methods=methods or ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"],
        rs_cpc_prototype_inits=prototype_inits or ["mean", "random_group_mean", "medoid"],
        rs_cpc_m_values=m_values or [1, 2, 4, 8],
        manifest_template=str(root / "outputs" / "manifests" / "feature_cache_after_seed{seed}_support" / "feature_cache_manifest.json"),
        text_cache_template=str(
            root
            / "outputs"
            / "features"
            / "{backbone}"
            / "{dataset}"
            / "base_seed{seed}"
            / "{dataset}"
            / "{backbone}"
            / "text"
            / "*"
            / "text_feature_cache.pt"
        ),
        adapter_plan_template=str(
            root
            / "outputs"
            / "preflight"
            / "adapter_input_plans"
            / "{dataset}_{backbone}_seed{seed}"
            / "*"
            / "adapter_input_plan.json"
        ),
        adapter_preflight_template=str(
            root
            / "outputs"
            / "preflight"
            / "adapter_input"
            / "{dataset}_{backbone}_seed{seed}"
            / "adapter_input_preflight_report.json"
        ),
        prototype_preflight_template=str(
            root
            / "outputs"
            / "preflight"
            / "rs_cpc_prototypes"
            / "{dataset}_{backbone}_seed{seed}"
            / "*"
            / "rs_cpc_prototype_preflight_report.json"
        ),
        output_dir=output_dir or root / "outputs" / "preflight" / "server_full_protocol",
        execution_env="remote_server",
        run_mode="local_validation",
        command="pytest server full protocol preflight",
    )


def write_complete_seed_artifacts(
    root: Path,
    *,
    seed: int,
    shots: list[int] | None = None,
    m_values: list[int] | None = None,
    prototype_inits: list[str] | None = None,
    text_dry_run: bool = False,
) -> None:
    shots = shots or [1, 2, 4, 8, 16]
    m_values = m_values or [1, 2, 4, 8]
    prototype_inits = prototype_inits or ["mean", "random_group_mean", "medoid"]
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    entries = []
    for section in ["val", "test"]:
        entries.append(write_cache_entry(root, dataset=dataset, backbone=backbone, seed=seed, section=section, split_id=f"base_seed{seed}"))
    for shot in shots:
        entries.append(write_cache_entry(root, dataset=dataset, backbone=backbone, seed=seed, section="support", split_id=f"shot_{shot}_seed{seed}", shot=shot))
    manifest_path = root / "outputs" / "manifests" / f"feature_cache_after_seed{seed}_support" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    write_text_cache(root, dataset=dataset, backbone=backbone, seed=seed, dry_run=text_dry_run)
    write_adapter_preflight(root, dataset=dataset, backbone=backbone, seed=seed, shots=shots)
    write_adapter_plan(root, dataset=dataset, backbone=backbone, seed=seed, shots=shots, m_values=m_values)
    write_prototype_preflight(root, dataset=dataset, backbone=backbone, seed=seed, shots=shots, m_values=m_values, prototype_inits=prototype_inits)


def write_cache_entry(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    seed: int,
    section: str,
    split_id: str,
    shot: int | None = None,
) -> dict[str, object]:
    cache_path = root / "outputs" / "features" / backbone / dataset / f"base_seed{seed}" / split_id / section / "feature_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"placeholder cache")
    return {
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "shot": shot,
        "split_id": split_id,
        "split_section": section,
        "feature_cache_path": str(cache_path),
        "run_dir": str(cache_path.parent),
    }


def write_text_cache(root: Path, *, dataset: str, backbone: str, seed: int, dry_run: bool) -> None:
    path = (
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "dataset": dataset,
                "backbone": backbone,
                "base_split": f"base_seed{seed}",
                "text_features": [[0.0] * 512 for _ in range(10)],
                "class_names": [f"class_{index}" for index in range(10)],
                "class_to_idx": {f"class_{index}": index for index in range(10)},
                "feature_dim": 512,
                "num_classes": 10,
                "dry_run": dry_run,
                "uses_fake_text_features": dry_run,
                "is_paper_result": False,
                "created_at": "2026-05-13T00:00:00+00:00",
            },
            handle,
        )


def write_adapter_preflight(root: Path, *, dataset: str, backbone: str, seed: int, shots: list[int]) -> None:
    per_split_summary: dict[str, object] = {
        f"base_seed{seed}": {
            "split_kind": "base",
            "sections": {"val": {"is_ready": True}, "test": {"is_ready": True}},
            "val_ready_for_tuning_input": True,
            "test_ready_for_evaluation_input": True,
        }
    }
    for shot in shots:
        per_split_summary[f"shot_{shot}_seed{seed}"] = {
            "split_kind": "shot",
            "shot": shot,
            "support": {"is_ready": True},
            "support_balanced": True,
            "min_support_per_class": shot,
        }
    path = root / "outputs" / "preflight" / "adapter_input" / f"{dataset}_{backbone}_seed{seed}" / "adapter_input_preflight_report.json"
    safe_write_json(
        path,
        {
            "is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "checked_methods": ["tip_adapter", "proto_adapter", "rs_cpc"],
            "per_split_summary": per_split_summary,
        },
    )


def write_adapter_plan(root: Path, *, dataset: str, backbone: str, seed: int, shots: list[int], m_values: list[int]) -> None:
    rows = []
    for shot in shots:
        for method in ["tip_adapter", "proto_adapter"]:
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": f"seed{seed}",
                    "shot_split": f"shot_{shot}_seed{seed}",
                    "shot": shot,
                    "method": method,
                    "candidate_M": None,
                    "is_ready": True,
                    "expected_cache_entries": 10 * shot if method == "tip_adapter" else 10,
                }
            )
        for m_value in m_values:
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "seed": f"seed{seed}",
                    "shot_split": f"shot_{shot}_seed{seed}",
                    "shot": shot,
                    "method": "rs_cpc",
                    "candidate_M": m_value,
                    "is_ready": m_value <= shot,
                    "expected_cache_entries": 10 * m_value,
                }
            )
    path = root / "outputs" / "preflight" / "adapter_input_plans" / f"{dataset}_{backbone}_seed{seed}" / "20260513T000000" / "adapter_input_plan.json"
    safe_write_json(
        path,
        {
            "source_preflight_is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "seed": f"seed{seed}",
            "rows": rows,
        },
    )


def write_prototype_preflight(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    seed: int,
    shots: list[int],
    m_values: list[int],
    prototype_inits: list[str],
) -> None:
    rows = []
    for shot in shots:
        for init_mode in prototype_inits:
            for m_value in m_values:
                if init_mode == "mean" and m_value != 1:
                    continue
                if m_value > shot:
                    continue
                rows.append(
                    {
                        "shot_split": f"shot_{shot}_seed{seed}",
                        "shot": shot,
                        "candidate_M": m_value,
                        "prototype_init": init_mode,
                        "is_ready": True,
                    }
                )
    path = root / "outputs" / "preflight" / "rs_cpc_prototypes" / f"{dataset}_{backbone}_seed{seed}" / "20260513T000000" / "rs_cpc_prototype_preflight_report.json"
    safe_write_json(
        path,
        {
            "is_valid": True,
            "dataset": dataset,
            "backbone": backbone,
            "seed": f"seed{seed}",
            "per_combination_summary": rows,
        },
    )


if __name__ == "__main__":
    unittest.main()
