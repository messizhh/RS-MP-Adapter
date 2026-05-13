from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_local_validation_results import summarize_local_validation_results
from src.utils.io import read_json, safe_write_json


class LocalValidationSummarizerTest(unittest.TestCase):
    def test_summarizes_fake_zero_shot_tip_proto_and_rs_cpc_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results" / "raw"
            write_fake_result_run(root, method="zero_shot", run_id="zero", top1_val=0.3, top1_test=0.4)
            write_fake_result_run(root, method="tip_adapter", shot=1, run_id="tip", top1_val=0.5, top1_test=0.6)
            write_fake_result_run(root, method="proto_adapter", shot=2, run_id="proto", top1_val=0.7, top1_test=0.8)
            write_fake_result_run(
                root,
                method="rs_cpc",
                shot=2,
                m_value=1,
                prototype_init="mean",
                run_id="rs_cpc",
                top1_val=0.9,
                top1_test=0.91,
                result_preflight_valid=True,
            )

            summary = summarize_local_validation_results(
                results_root=results_root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                seed=1,
                output_dir=root / "outputs" / "analysis" / "local_validation_summaries",
                include_methods=["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"],
                command="pytest summarize local validation",
            )

            rows = summary["rows"]
            self.assertEqual([row["method"] for row in rows], ["zero_shot", "tip_adapter", "proto_adapter", "rs_cpc"])
            self.assertEqual(rows[0]["test_top1_acc"], 0.4)
            self.assertEqual(rows[3]["M"], 1)
            self.assertEqual(rows[3]["prototype_init"], "mean")
            self.assertEqual(rows[3]["result_preflight_status"], "valid=true")
            self.assertTrue(Path(summary["csv_path"]).exists())
            self.assertTrue(Path(summary["markdown_path"]).exists())
            payload = read_json(summary["json_path"])
            self.assertFalse(payload["is_paper_result"])
            self.assertFalse(payload["writes_results_raw"])
            self.assertFalse(payload["computes_logits"])
            self.assertFalse(payload["computes_accuracy"])
            self.assertFalse(payload["evaluates_model"])
            self.assertFalse(payload["trains_model"])
            self.assertFalse(payload["modifies_results"])
            self.assertFalse(payload["deletes_results"])
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")
            self.assertIn("This is local_validation only.", markdown)
            self.assertIn("Not eligible for paper tables.", markdown)
            self.assertIn("Do not cite as final result.", markdown)
            with Path(summary["csv_path"]).open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(csv_rows[3]["run_id"], "rs_cpc")

    def test_multiple_runs_for_same_combo_selects_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results" / "raw"
            write_fake_result_run(
                root,
                method="tip_adapter",
                shot=1,
                run_id="old_run",
                top1_test=0.2,
                end_time="2026-05-12T00:00:00+00:00",
            )
            write_fake_result_run(
                root,
                method="tip_adapter",
                shot=1,
                run_id="new_run",
                top1_test=0.9,
                end_time="2026-05-12T00:01:00+00:00",
            )

            summary = summarize_local_validation_results(
                results_root=results_root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                seed=1,
                output_dir=root / "outputs" / "analysis",
                include_methods=["tip_adapter"],
                command="pytest latest run",
            )

            self.assertEqual(len(summary["rows"]), 1)
            row = summary["rows"][0]
            self.assertEqual(row["run_id"], "new_run")
            self.assertEqual(row["test_top1_acc"], 0.9)
            self.assertEqual(row["num_candidate_runs"], 2)

    def test_paper_result_or_paper_table_eligible_runs_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results" / "raw"
            write_fake_result_run(root, method="zero_shot", run_id="paper", is_paper_result=True)
            write_fake_result_run(root, method="tip_adapter", shot=1, run_id="eligible", eligible_for_paper_tables=True)
            write_fake_result_run(root, method="proto_adapter", shot=1, run_id="local_only")

            summary = summarize_local_validation_results(
                results_root=results_root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                seed=1,
                output_dir=root / "outputs" / "analysis",
                include_methods=["zero_shot", "tip_adapter", "proto_adapter"],
                command="pytest exclude paper flags",
            )

            self.assertEqual([row["run_id"] for row in summary["rows"]], ["local_only"])
            payload = read_json(summary["json_path"])
            self.assertEqual(payload["excluded_reason_counts"]["is_paper_result_not_false"], 1)
            self.assertEqual(payload["excluded_reason_counts"]["eligible_for_paper_tables_not_false"], 1)

    def test_mismatched_dataset_backbone_or_seed_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results" / "raw"
            write_fake_result_run(root, method="zero_shot", run_id="valid")
            write_fake_result_run(root, method="zero_shot", dataset="aid", run_id="wrong_dataset")
            write_fake_result_run(root, method="zero_shot", backbone="clip_vit_b16", run_id="wrong_backbone")
            write_fake_result_run(root, method="zero_shot", seed=2, run_id="wrong_seed")

            summary = summarize_local_validation_results(
                results_root=results_root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                seed=1,
                output_dir=root / "outputs" / "analysis",
                include_methods=["zero_shot"],
                command="pytest exclude mismatches",
            )

            self.assertEqual(len(summary["rows"]), 1)
            self.assertEqual(summary["rows"][0]["run_id"], "valid")
            payload = read_json(summary["json_path"])
            self.assertEqual(payload["excluded_reason_counts"]["dataset_mismatch"], 1)
            self.assertEqual(payload["excluded_reason_counts"]["backbone_mismatch"], 1)
            self.assertEqual(payload["excluded_reason_counts"]["seed_mismatch"], 1)

    def test_summarizer_does_not_modify_existing_metrics_or_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results" / "raw"
            run_dir = write_fake_result_run(root, method="rs_cpc", shot=4, m_value=2, prototype_init="random_group_mean")
            metrics_before = (run_dir / "metrics.json").read_bytes()
            metadata_before = (run_dir / "metadata.json").read_bytes()

            summary = summarize_local_validation_results(
                results_root=results_root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                seed=1,
                output_dir=root / "outputs" / "analysis",
                include_methods=["rs_cpc"],
                command="pytest readonly summarizer",
            )

            self.assertEqual(len(summary["rows"]), 1)
            self.assertEqual((run_dir / "metrics.json").read_bytes(), metrics_before)
            self.assertEqual((run_dir / "metadata.json").read_bytes(), metadata_before)


def write_fake_result_run(
    root: Path,
    *,
    method: str,
    dataset: str = "eurosat",
    backbone: str = "remoteclip_vit_b32",
    seed: int = 1,
    shot: int | None = None,
    m_value: int | None = None,
    prototype_init: str | None = None,
    run_id: str = "run",
    top1_val: float = 0.5,
    top1_test: float = 0.6,
    end_time: str = "2026-05-12T00:00:00+00:00",
    is_paper_result: bool = False,
    eligible_for_paper_tables: bool = False,
    result_preflight_valid: bool = False,
) -> Path:
    run_dir = fake_run_dir(
        root,
        dataset=dataset,
        backbone=backbone,
        method=method,
        seed=seed,
        shot=shot,
        m_value=m_value,
        prototype_init=prototype_init,
        run_id=run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if result_preflight_valid:
        safe_write_json(run_dir / "result_run_preflight_report.json", {"is_valid": True})
    metadata = base_payload(
        run_dir=run_dir,
        method=method,
        dataset=dataset,
        backbone=backbone,
        seed=seed,
        shot=shot,
        m_value=m_value,
        prototype_init=prototype_init,
        run_id=run_id,
        top1_val=top1_val,
        top1_test=top1_test,
        end_time=end_time,
        is_paper_result=is_paper_result,
        eligible_for_paper_tables=eligible_for_paper_tables,
    )
    metrics = dict(metadata)
    metrics.update(
        {
            "top1_acc": top1_test,
            "top1_acc_by_split": {"val": top1_val, "test": top1_test},
            "per_split": {"val": {"top1_acc": top1_val}, "test": {"top1_acc": top1_test}},
            "cache_entries": cache_entries(method=method, shot=shot, m_value=m_value),
            "trainable_params": 0,
            "training_time_sec": 0.0,
            "inference_time_sec": 0.1,
            "images_per_second": 10.0,
            "gpu_memory_mb": None,
        }
    )
    safe_write_json(run_dir / "metadata.json", metadata)
    safe_write_json(run_dir / "metrics.json", metrics)
    return run_dir


def fake_run_dir(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    method: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str | None,
    run_id: str,
) -> Path:
    base = root / "results" / "raw" / dataset / backbone / method
    if method == "zero_shot":
        return base / f"seed_{seed}" / run_id
    if method == "rs_cpc":
        return base / f"shot_{shot}" / f"M_{m_value}" / str(prototype_init) / f"seed_{seed}" / run_id
    return base / f"shot_{shot}" / f"seed_{seed}" / run_id


def base_payload(
    *,
    run_dir: Path,
    method: str,
    dataset: str,
    backbone: str,
    seed: int,
    shot: int | None,
    m_value: int | None,
    prototype_init: str | None,
    run_id: str,
    top1_val: float,
    top1_test: float,
    end_time: str,
    is_paper_result: bool,
    eligible_for_paper_tables: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "method": method,
        "dataset": dataset,
        "backbone": backbone,
        "seed": seed,
        "shot": shot,
        "execution_env": "local_wsl",
        "run_mode": "local_validation",
        "is_paper_result": is_paper_result,
        "eligible_for_paper_tables": eligible_for_paper_tables,
        "cache_entries": cache_entries(method=method, shot=shot, m_value=m_value),
        "top1_acc_by_split": {"val": top1_val, "test": top1_test},
        "result_json_path": str(run_dir / "metrics.json"),
        "log_path": str(run_dir / "log.txt"),
        "start_time": "2026-05-12T00:00:00+00:00",
        "end_time": end_time,
    }
    if method == "rs_cpc":
        payload["M"] = m_value
        payload["prototype_init"] = prototype_init
    return payload


def cache_entries(*, method: str, shot: int | None, m_value: int | None) -> int:
    num_classes = 2
    if method == "tip_adapter":
        return num_classes * int(shot or 0)
    if method == "rs_cpc":
        return num_classes * int(m_value or 0)
    return num_classes


if __name__ == "__main__":
    unittest.main()
