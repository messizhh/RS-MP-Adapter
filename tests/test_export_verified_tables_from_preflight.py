from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.export_verified_tables_from_preflight import export_verified_tables_from_preflight
from src.utils.io import read_json, safe_write_json


class ExportVerifiedTablesFromPreflightTest(unittest.TestCase):
    def test_includes_verified_server_full_with_raw_paper_flags_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_run(
                root,
                method="rs_cpc",
                run_mode="server_full",
                execution_env="remote_server",
                shot=4,
                seed=1,
                m_value=2,
                prototype_init="random_group_mean",
                top1_acc=0.7,
            )
            summary_path = write_preflight_summary(root, [run_dir])

            result = export_verified_tables_from_preflight(
                preflight_summary=summary_path,
                output_root=root / "tables",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                command="unit test verified export",
            )

            self.assertEqual(result["num_included"], 1)
            registry_rows = read_csv(Path(result["output_dir"]) / "inclusion_registry.csv")
            self.assertEqual(len(registry_rows), 1)
            self.assertEqual(registry_rows[0]["num_prototypes_per_class"], "2")
            self.assertEqual(registry_rows[0]["prototype_init"], "random_group_mean")
            self.assertEqual(registry_rows[0]["raw_metrics_is_paper_result"], "False")
            self.assertEqual(registry_rows[0]["raw_metrics_eligible_for_paper_tables"], "False")
            self.assertEqual(registry_rows[0]["paper_facing_policy_status"], "pending_final_policy")

            audit = read_json(Path(result["output_dir"]) / "day2_table_audit_summary.json")
            self.assertEqual(audit["raw_paper_flag_counts"]["metrics_is_paper_result_false"], 1)
            self.assertEqual(audit["raw_paper_flag_counts"]["metrics_eligible_for_paper_tables_false"], 1)
            self.assertEqual(audit["paper_facing_status"], "not_marked_as_final_paper_tables")
            self.assertTrue((Path(result["output_dir"]) / "per_class_confusion_not_available.md").exists())

    def test_excludes_local_debug_smoke_modes_and_does_not_scan_unlisted_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            included = write_fake_run(
                root,
                method="tip_adapter",
                run_mode="server_full",
                execution_env="remote_server",
                shot=1,
                seed=1,
                top1_acc=0.6,
            )
            excluded_modes = ["dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"]
            excluded_runs = [
                write_fake_run(
                    root,
                    method="tip_adapter",
                    run_mode=mode,
                    execution_env="local_wsl" if mode != "debug" else "remote_server",
                    shot=1,
                    seed=index + 2,
                    top1_acc=0.9,
                )
                for index, mode in enumerate(excluded_modes)
            ]
            unlisted = write_fake_run(
                root,
                method="tip_adapter",
                run_mode="server_full",
                execution_env="remote_server",
                shot=1,
                seed=99,
                top1_acc=0.99,
                run_id="unlisted",
            )
            self.assertTrue((unlisted / "metrics.json").exists())
            summary_path = write_preflight_summary(root, [included, *excluded_runs])

            result = export_verified_tables_from_preflight(
                preflight_summary=summary_path,
                output_root=root / "tables",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                command="unit test verified export",
            )

            self.assertEqual(result["num_included"], 1)
            self.assertEqual(result["num_excluded"], len(excluded_modes))
            seed_rows = read_csv(Path(result["output_dir"]) / "main_accuracy_seed_rows.csv")
            self.assertEqual([row["seed"] for row in seed_rows], ["1"])
            self.assertNotIn("99", [row["seed"] for row in seed_rows])
            audit = read_json(Path(result["output_dir"]) / "day2_table_audit_summary.json")
            self.assertTrue(audit["does_not_scan_raw_root"])
            for mode in excluded_modes:
                self.assertTrue(any(mode in reason for reason in audit["exclusion_reason_counts"]))

    def test_conflicting_rs_cpc_m_values_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_fake_run(
                root,
                method="rs_cpc",
                run_mode="server_full",
                execution_env="remote_server",
                shot=4,
                seed=1,
                m_value=2,
                prototype_init="random_group_mean",
                top1_acc=0.7,
                run_m_dir=4,
            )
            summary_path = write_preflight_summary(root, [run_dir])

            result = export_verified_tables_from_preflight(
                preflight_summary=summary_path,
                output_root=root / "tables",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                command="unit test verified export",
            )

            self.assertEqual(result["num_included"], 0)
            registry = read_json(Path(result["output_dir"]) / "inclusion_registry.json")
            self.assertIn("conflicting RS-CPC M values", registry["excluded"][0]["exclusion_reasons"])


def write_preflight_summary(root: Path, run_dirs: list[Path]) -> Path:
    path = root / "outputs" / "preflight" / "post_run_preflight_summary.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["run_dir", "report_path", "preflight_exit_code", "is_valid"],
        )
        writer.writeheader()
        for run_dir in run_dirs:
            writer.writerow(
                {
                    "run_dir": str(run_dir),
                    "report_path": str(run_dir / "result_run_preflight_report.json"),
                    "preflight_exit_code": 0,
                    "is_valid": True,
                }
            )
    return path


def write_fake_run(
    root: Path,
    *,
    method: str,
    run_mode: str,
    execution_env: str,
    shot: int,
    seed: int,
    top1_acc: float,
    m_value: int | None = None,
    prototype_init: str | None = None,
    run_m_dir: int | None = None,
    run_id: str = "run1",
) -> Path:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    if method == "rs_cpc":
        m_dir = run_m_dir if run_m_dir is not None else m_value
        run_dir = (
            root
            / "results"
            / "raw"
            / dataset
            / backbone
            / method
            / f"shot_{shot}"
            / f"M_{m_dir}"
            / str(prototype_init)
            / f"seed_{seed}"
            / run_id
        )
    else:
        run_dir = root / "results" / "raw" / dataset / backbone / method / f"shot_{shot}" / f"seed_{seed}" / run_id
    command = f"python3 scripts/run_{method}.py --dataset {dataset} --backbone {backbone} --shot {shot}"
    cache_entries = 10 * shot
    if method == "rs_cpc":
        command += f" --M {m_value} --prototype-init {prototype_init}"
        cache_entries = 10 * int(m_value or 0)
    metrics = {
        "run_id": run_id,
        "method": method,
        "backbone": backbone,
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "eligible_for_paper_tables": False,
        "device": "cuda" if execution_env == "remote_server" else "cpu",
        "top1_acc": top1_acc,
        "num_classes": 10,
        "cache_entries": cache_entries,
        "trainable_params": 0,
        "training_time_sec": 0.0,
        "inference_time_sec": 1.0,
        "images_per_second": 10.0,
        "gpu_memory_mb": None,
        "uses_fake_data": False,
        "uses_fake_features": False,
        "fake_or_dry_run": False,
        "command": command,
    }
    metadata = {
        **metrics,
        "command": command,
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(run_dir / "log.txt"),
    }
    if method != "rs_cpc":
        metrics["compression_ratio"] = ""
    safe_write_json(run_dir / "metrics.json", metrics)
    safe_write_json(run_dir / "metadata.json", metadata)
    (run_dir / "log.txt").write_text("fake completed run\n", encoding="utf-8")
    safe_write_json(run_dir / "result_run_preflight_report.json", {"is_valid": True, "run_dir": str(run_dir)})
    return run_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
