from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.check_adapter_input_preflight import run_adapter_input_preflight
from src.utils.io import read_json, safe_write_json


class AdapterInputPreflightTest(unittest.TestCase):
    def test_fake_cache_manifest_passes_adapter_input_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_fake_preflight_case(root, shots=[8])

            report_path, is_valid = run_adapter_input_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                shot_splits=[str(case["shot_split_paths"][8])],
                methods=["tip_adapter", "proto_adapter", "rs_cpc"],
                output_dir=root / "preflight" / "adapter_input",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest fake adapter input preflight",
            )

            report = read_json(report_path)
            shot_key = str(case["shot_split_paths"][8])
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            self.assertEqual(report["feature_dim"], 512)
            self.assertEqual(report["num_classes"], 3)
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertTrue(report["per_split_summary"][str(case["base_split_path"])]["val_ready_for_tuning_input"])
            self.assertTrue(report["per_split_summary"][str(case["base_split_path"])]["test_ready_for_evaluation_input"])
            self.assertEqual(
                report["per_method_input_summary"]["tip_adapter"]["per_shot"][shot_key]["expected_cache_entries"],
                24,
            )
            self.assertEqual(
                report["per_method_input_summary"]["proto_adapter"]["per_shot"][shot_key]["expected_cache_entries"],
                3,
            )
            self.assertEqual(
                report["per_method_input_summary"]["rs_cpc"]["per_shot"][shot_key]["expected_cache_entries_by_M"],
                {"1": 3, "2": 6, "4": 12, "8": 24},
            )
            self.assertTrue(report["per_method_input_summary"]["rs_cpc"]["per_shot"][shot_key]["method_input_ready"])
            self.assertEqual(
                report_path,
                root
                / "preflight"
                / "adapter_input"
                / "eurosat_remoteclip_vit_b32_seed1"
                / "adapter_input_preflight_report.json",
            )

    def test_missing_cache_file_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_fake_preflight_case(root, shots=[2])
            missing_cache = case["support_cache_paths"][2]
            missing_cache.unlink()

            report_path, is_valid = run_adapter_input_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                shot_splits=[str(case["shot_split_paths"][2])],
                methods=["tip_adapter"],
                output_dir=root / "preflight",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest missing adapter cache",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(any("feature cache file does not exist" in error for error in report["errors"]))

    def test_feature_dim_inconsistency_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_fake_preflight_case(root, shots=[2], support_feature_dims={2: 256})

            report_path, is_valid = run_adapter_input_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                shot_splits=[str(case["shot_split_paths"][2])],
                methods=["proto_adapter"],
                output_dir=root / "preflight",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest inconsistent adapter feature dim",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(any("feature_dim is inconsistent" in error for error in report["errors"]))

    def test_rs_cpc_m_larger_than_shot_warns_and_marks_m_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = make_fake_preflight_case(root, shots=[1])

            report_path, is_valid = run_adapter_input_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                shot_splits=[str(case["shot_split_paths"][1])],
                methods=["rs_cpc"],
                output_dir=root / "preflight",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest rs cpc m greater than shot",
            )

            report = read_json(report_path)
            shot_key = str(case["shot_split_paths"][1])
            rs_cpc = report["per_method_input_summary"]["rs_cpc"]["per_shot"][shot_key]
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertTrue(any("M=2 exceeds" in warning for warning in report["warnings"]))
            self.assertTrue(rs_cpc["method_input_ready_by_M"]["1"])
            self.assertFalse(rs_cpc["method_input_ready_by_M"]["2"])
            self.assertFalse(rs_cpc["method_input_ready"])


def make_fake_preflight_case(
    root: Path,
    *,
    shots: list[int],
    support_feature_dims: dict[int, int] | None = None,
) -> dict[str, object]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 3
    entries = []
    support_feature_dims = support_feature_dims or {}
    base_split_path = write_split(root, dataset, "base_seed1", num_classes, shot=None)
    for section, count in [("train", 12), ("val", 6), ("test", 6)]:
        entries.append(
            write_summary_and_cache(
                root,
                dataset=dataset,
                backbone=backbone,
                split_id="base_seed1",
                split_path=base_split_path,
                section=section,
                labels=[idx % num_classes for idx in range(count)],
                feature_dim=512,
                num_classes=num_classes,
            )
        )

    shot_split_paths: dict[int, Path] = {}
    support_cache_paths: dict[int, Path] = {}
    for shot in shots:
        split_id = f"shot_{shot}_seed1"
        split_path = write_split(root, dataset, split_id, num_classes, shot=shot)
        shot_split_paths[shot] = split_path
        labels = [label for label in range(num_classes) for _ in range(shot)]
        entry = write_summary_and_cache(
            root,
            dataset=dataset,
            backbone=backbone,
            split_id=split_id,
            split_path=split_path,
            section="support",
            labels=labels,
            feature_dim=support_feature_dims.get(shot, 512),
            num_classes=num_classes,
        )
        entries.append(entry)
        support_cache_paths[shot] = Path(entry["feature_cache_path"])

    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    return {
        "manifest_path": manifest_path,
        "base_split_path": base_split_path,
        "shot_split_paths": shot_split_paths,
        "support_cache_paths": support_cache_paths,
    }


def write_split(root: Path, dataset: str, split_id: str, num_classes: int, shot: int | None) -> Path:
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    train = make_rows(12, num_classes)
    val = make_rows(6, num_classes)
    test = make_rows(6, num_classes)
    support = [row for label in range(num_classes) for row in make_rows(shot or 0, num_classes, fixed_label=label)]
    path = root / "splits" / dataset / f"{split_id}.json"
    safe_write_json(
        path,
        {
            "dataset": dataset,
            "seed": 1,
            "shot": shot,
            "train": train,
            "val": val,
            "test": test,
            "support": support,
            "class_to_idx": class_to_idx,
            "num_classes": num_classes,
        },
    )
    return path


def make_rows(count: int, num_classes: int, fixed_label: int | None = None) -> list[dict[str, object]]:
    rows = []
    for idx in range(count):
        label = fixed_label if fixed_label is not None else idx % num_classes
        rows.append({"class_name": f"class_{label}", "label": label, "path": f"class_{label}/image_{idx}.jpg"})
    return rows


def write_summary_and_cache(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    split_id: str,
    split_path: Path,
    section: str,
    labels: list[int],
    feature_dim: int,
    num_classes: int,
) -> dict[str, str]:
    run_dir = root / "features" / backbone / dataset / split_id / section / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "feature_cache.pt"
    summary_path = run_dir / "feature_extraction_summary.json"
    features = [[float(label) for _ in range(feature_dim)] for label in labels]
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "image_features": features,
                "image_labels": labels,
                "image_paths": [f"fake://{split_id}/{section}/{idx}.jpg" for idx in range(len(labels))],
                "split_name": section,
                "class_to_idx": {f"class_{idx}": idx for idx in range(num_classes)},
                "backbone": backbone,
                "dataset": dataset,
                "feature_dim": feature_dim,
                "normalize_features": True,
                "created_at": "2026-05-12T00:00:00+00:00",
                "source_script": "tests/test_adapter_input_preflight.py",
                "metadata": {"dataset": dataset, "backbone": backbone},
            },
            handle,
        )
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": str(split_path),
            "split_section": section,
            "image_count": len(labels),
            "feature_shape": [len(labels), feature_dim],
            "feature_cache_path": str(cache_path),
            "run_dir": str(run_dir),
            "git_commit": "abc123",
            "checkpoint_loaded": True,
            "is_paper_result": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "saves_predictions": False,
            "saves_logits": False,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(cache_path), "run_dir": str(run_dir)}


if __name__ == "__main__":
    unittest.main()
