from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_feature_cache_manifest import build_feature_cache_manifest
from src.utils.io import read_json, safe_write_json


class FeatureCacheManifestTest(unittest.TestCase):
    def test_manifest_summarizes_feature_extraction_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_root = root / "features"
            write_summary(
                features_root / "remoteclip_vit_b32" / "eurosat" / "train" / "run1",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_section="train",
                image_count=100,
                checkpoint_loaded=True,
            )
            write_summary(
                features_root / "remoteclip_vit_b32" / "eurosat" / "val" / "run1",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_section="val",
                image_count=20,
                checkpoint_loaded=True,
            )

            result = build_feature_cache_manifest(
                features_root=features_root,
                output_dir=root / "manifest",
                execution_env="local_wsl",
                run_mode="local_validation",
            )

            manifest = read_json(result["manifest_json_path"])
            summary = read_json(result["manifest_summary_path"])
            self.assertEqual(len(manifest["entries"]), 2)
            self.assertEqual(summary["num_entries"], 2)
            self.assertEqual(summary["datasets"], ["eurosat"])
            self.assertEqual(summary["backbones"], ["remoteclip_vit_b32"])
            self.assertEqual(summary["split_sections"], ["train", "val"])
            self.assertEqual(summary["total_images"], 120)
            self.assertEqual(summary["num_paper_results"], 0)
            self.assertEqual(summary["num_eligible_for_paper_tables"], 0)
            self.assertEqual(summary["num_with_checkpoint_loaded_false"], 0)
            self.assertFalse(summary["manifest_is_paper_result"])
            self.assertTrue(summary["reads_feature_extraction_summary_only"])
            self.assertFalse(summary["loads_model"])
            self.assertFalse(summary["extracts_features"])
            self.assertFalse(summary["trains_model"])
            self.assertFalse(summary["evaluates_model"])

            with result["manifest_csv_path"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset"], "eurosat")
            self.assertIn("feature_cache_path", rows[0])

    def test_manifest_counts_and_lists_warning_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_root = root / "features"
            write_summary(
                features_root / "bad" / "run1",
                dataset="aid",
                backbone="remoteclip_vit_b32",
                split_section="test",
                image_count=5,
                checkpoint_loaded=False,
                trains_model=True,
                evaluates_model=True,
                saves_predictions=True,
                is_paper_result=True,
                eligible_for_paper_tables=True,
                extracts_text_features=True,
            )

            result = build_feature_cache_manifest(
                features_root=features_root,
                output_dir=root / "manifest",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            summary = read_json(result["manifest_summary_path"])
            self.assertEqual(summary["num_entries"], 1)
            self.assertEqual(summary["num_paper_results"], 1)
            self.assertEqual(summary["num_eligible_for_paper_tables"], 1)
            self.assertEqual(summary["num_with_checkpoint_loaded_false"], 1)
            self.assertEqual(summary["num_with_training_true"], 1)
            self.assertEqual(summary["num_with_evaluation_true"], 1)
            self.assertEqual(summary["num_with_predictions_true"], 1)
            self.assertEqual(summary["num_with_text_features_true"], 1)
            self.assertEqual(len(summary["warning_entries"]), 1)
            self.assertEqual(
                sorted(summary["warning_entries"][0]["warning_flags"]),
                sorted(["trains_model", "evaluates_model", "saves_predictions", "is_paper_result", "eligible_for_paper_tables"]),
            )

    def test_manifest_preserves_seed_shot_and_split_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_root = root / "features"
            write_summary(
                features_root / "remoteclip_vit_b32" / "eurosat" / "support" / "run1",
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_section="support",
                image_count=10,
                checkpoint_loaded=True,
                seed=2,
                shot=1,
                split="splits/eurosat/shot_1_seed2.json",
                split_id="shot_1_seed2",
                split_name="shot_1_seed2",
            )

            result = build_feature_cache_manifest(
                features_root=features_root,
                output_dir=root / "manifest",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            manifest = read_json(result["manifest_json_path"])
            entry = manifest["entries"][0]
            self.assertEqual(entry["seed"], 2)
            self.assertEqual(entry["shot"], 1)
            self.assertEqual(entry["split"], "splits/eurosat/shot_1_seed2.json")
            self.assertEqual(entry["split_id"], "shot_1_seed2")
            self.assertEqual(entry["split_name"], "shot_1_seed2")
            self.assertEqual(entry["split_path"], "splits/eurosat/shot_1_seed2.json")
            self.assertEqual(entry["num_samples"], 10)

    def test_manifest_cli_writes_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_root = root / "features"
            write_summary(
                features_root / "remoteclip_vit_b32" / "nwpu_resisc45" / "test" / "run1",
                dataset="nwpu_resisc45",
                backbone="remoteclip_vit_b32",
                split_section="test",
                image_count=10,
                checkpoint_loaded=True,
            )
            output_dir = root / "manifest"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_feature_cache_manifest.py",
                    "--features-root",
                    str(features_root),
                    "--output-dir",
                    str(output_dir),
                    "--execution-env",
                    "local_wsl",
                    "--run-mode",
                    "local_validation",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_summary_path = extract_path(completed.stdout, "manifest_summary_path")
            self.assertEqual(manifest_summary_path, output_dir / "feature_cache_manifest_summary.json")
            self.assertTrue((output_dir / "feature_cache_manifest.json").exists())
            self.assertTrue((output_dir / "feature_cache_manifest.csv").exists())


def write_summary(
    run_dir: Path,
    *,
    dataset: str,
    backbone: str,
    split_section: str,
    image_count: int,
    checkpoint_loaded: bool,
    trains_model: bool = False,
    evaluates_model: bool = False,
    saves_predictions: bool = False,
    is_paper_result: bool = False,
    eligible_for_paper_tables: bool = False,
    extracts_text_features: bool = False,
    seed: int | None = None,
    shot: int | None = None,
    split: str | None = None,
    split_id: str | None = None,
    split_name: str | None = None,
) -> Path:
    path = run_dir / "feature_extraction_summary.json"
    split_path = split or f"splits/{dataset}/base_split_seed1.json"
    safe_write_json(
        path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "shot": shot,
            "split": split_path,
            "split_id": split_id,
            "split_name": split_name,
            "split_path": split_path,
            "split_section": split_section,
            "image_count": image_count,
            "num_samples": image_count,
            "feature_shape": [image_count, 512],
            "feature_cache_path": str(run_dir / "feature_cache.pt"),
            "run_dir": str(run_dir),
            "git_commit": "abc123",
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_load_mode": "direct_state_dict",
            "checkpoint_num_tensors": 302 if checkpoint_loaded else 0,
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "final_weights_loaded_from_checkpoint": checkpoint_loaded,
            "final_weight_source": "cli_override_checkpoint" if checkpoint_loaded else None,
            "final_checkpoint_load_status": "loaded_strictly_matching_keys" if checkpoint_loaded else "not_attempted",
            "is_real_feature_extraction": True,
            "is_full_feature_extraction": True,
            "is_limited_real_extraction": False,
            "is_paper_result": is_paper_result,
            "is_paper_result_candidate": False,
            "eligible_for_paper_tables": eligible_for_paper_tables,
            "trains_model": trains_model,
            "evaluates_model": evaluates_model,
            "extracts_text_features": extracts_text_features,
            "saves_predictions": saves_predictions,
            "saves_logits": False,
            "start_time": "2026-05-07T00:00:00+00:00",
            "end_time": "2026-05-07T00:01:00+00:00",
            "total_time_sec": 60.0,
        },
    )
    return path


def extract_path(stdout: str, key: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing {key} in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
