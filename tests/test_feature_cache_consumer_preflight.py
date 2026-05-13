from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_feature_cache_consumer_preflight import run_feature_cache_consumer_preflight
from src.utils.io import read_json, safe_write_json


class FeatureCacheConsumerPreflightTest(unittest.TestCase):
    def test_consumer_preflight_validates_base_and_shot_support_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = make_manifest(
                root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                num_classes=3,
                base_split="base_seed1",
                shot_splits=["shot_1_seed1", "shot_2_seed1"],
            )

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                shot_splits=["shot_1_seed1", "shot_2_seed1"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["required_base_sections_found"], ["train", "val", "test"])
            self.assertEqual(report["required_support_splits_found"], ["shot_1_seed1", "shot_2_seed1"])
            self.assertEqual(report["num_entries_checked"], 5)
            self.assertEqual(report["feature_dim"], 512)
            self.assertEqual(report["num_classes"], 3)
            self.assertEqual(report["support_counts_by_shot"], {"shot_1_seed1": 3, "shot_2_seed1": 6})
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["eligible_for_paper_tables"])

    def test_consumer_preflight_reports_invalid_cache_metadata_without_modifying_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = make_manifest(
                root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                num_classes=3,
                base_split="base_seed1",
                shot_splits=["shot_1_seed1"],
            )
            bad_summary = root / "features" / "remoteclip_vit_b32" / "eurosat" / "shot_1_seed1" / "support" / "run" / "feature_extraction_summary.json"
            data = read_json(bad_summary)
            data["image_count"] = 2
            data["feature_shape"] = [2, 512]
            data["saves_predictions"] = True
            safe_write_json(bad_summary, data, overwrite=True)

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                shot_splits=["shot_1_seed1"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(any("saves_predictions" in error for error in report["errors"]))
            self.assertTrue(any("shot_1_seed1 support image_count=2" in error for error in report["errors"]))
            self.assertTrue(bad_summary.exists())

    def test_consumer_preflight_allows_missing_final_weight_field_for_older_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = make_manifest(
                root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                num_classes=2,
                base_split="base_seed1",
                shot_splits=["shot_1_seed1"],
            )
            support_summary = root / "features" / "remoteclip_vit_b32" / "eurosat" / "shot_1_seed1" / "support" / "run" / "feature_extraction_summary.json"
            data = read_json(support_summary)
            data.pop("final_weights_loaded_from_checkpoint")
            safe_write_json(support_summary, data, overwrite=True)

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed1",
                shot_splits=["shot_1_seed1"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])

    def test_consumer_preflight_matches_explicit_seed2_split_metadata_without_path_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = "eurosat"
            backbone = "remoteclip_vit_b32"
            num_classes = 2
            entries = []
            for section, count in [("train", 8), ("val", 4), ("test", 4)]:
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id="base_seed2",
                        split_name="base_seed2",
                        split_path="splits/eurosat/base_split_seed2.json",
                        split_section=section,
                        image_count=count,
                        num_classes=num_classes,
                        run_key=f"base_{section}",
                        seed=2,
                        shot=None,
                    )
                )
            for shot in [1, 2, 4, 8, 16]:
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id=f"shot_{shot}_seed2",
                        split_name=f"shot_{shot}_seed2",
                        split_path=f"splits/eurosat/shot_{shot}_seed2.json",
                        split_section="support",
                        image_count=shot * num_classes,
                        num_classes=num_classes,
                        run_key=f"support_{shot}",
                        seed=2,
                        shot=shot,
                    )
                )
            manifest_path = root / "manifest" / "feature_cache_manifest.json"
            safe_write_json(manifest_path, {"entries": entries})

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset=dataset,
                backbone=backbone,
                base_split="base_seed2",
                shot_splits=["shot_1_seed2", "shot_2_seed2", "shot_4_seed2", "shot_8_seed2", "shot_16_seed2"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["required_base_sections_found"], ["train", "val", "test"])
            self.assertEqual(
                report["required_support_splits_found"],
                ["shot_1_seed2", "shot_2_seed2", "shot_4_seed2", "shot_8_seed2", "shot_16_seed2"],
            )
            self.assertEqual(
                report["support_counts_by_shot"],
                {
                    "shot_1_seed2": 2,
                    "shot_2_seed2": 4,
                    "shot_4_seed2": 8,
                    "shot_8_seed2": 16,
                    "shot_16_seed2": 32,
                },
            )
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["eligible_for_paper_tables"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])

    def test_consumer_preflight_does_not_double_count_duplicate_seed2_support_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = "eurosat"
            backbone = "remoteclip_vit_b32"
            num_classes = 10
            entries = []
            for section, count in [("train", 20), ("val", 10), ("test", 10)]:
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id=None,
                        split_name=None,
                        split_path="splits/eurosat/base_split_seed2.json",
                        split_section=section,
                        image_count=count,
                        num_classes=num_classes,
                        run_key=f"old_base_{section}",
                        seed=None,
                        shot=None,
                        start_time="2026-05-12T00:00:00+00:00",
                        end_time="2026-05-12T00:01:00+00:00",
                    )
                )
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id="base_seed2",
                        split_name="base_seed2",
                        split_path="splits/eurosat/base_split_seed2.json",
                        split_section=section,
                        image_count=count,
                        num_classes=num_classes,
                        run_key=f"new_base_{section}",
                        seed=2,
                        shot=None,
                        base_split="base_seed2",
                        start_time="2026-05-13T00:00:00+00:00",
                        end_time="2026-05-13T00:01:00+00:00",
                    )
                )
            for shot in [1, 2, 4, 8, 16]:
                split_id = f"shot_{shot}_seed2"
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id=None,
                        split_name=None,
                        split_path=f"splits/eurosat/{split_id}.json",
                        split_section="support",
                        image_count=shot * num_classes,
                        num_classes=num_classes,
                        run_key=f"old_support_{shot}",
                        seed=None,
                        shot=None,
                        start_time="2026-05-12T00:00:00+00:00",
                        end_time="2026-05-12T00:01:00+00:00",
                    )
                )
                entries.append(
                    write_generic_summary_and_cache(
                        root,
                        dataset=dataset,
                        backbone=backbone,
                        split_id=split_id,
                        split_name=split_id,
                        split_path=f"splits/eurosat/{split_id}.json",
                        split_section="support",
                        image_count=shot * num_classes,
                        num_classes=num_classes,
                        run_key=f"new_support_{shot}",
                        seed=2,
                        shot=shot,
                        start_time="2026-05-13T00:00:00+00:00",
                        end_time="2026-05-13T00:01:00+00:00",
                    )
                )
            manifest_path = root / "manifest" / "feature_cache_manifest.json"
            safe_write_json(manifest_path, {"entries": entries})

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset=dataset,
                backbone=backbone,
                base_split="base_seed2",
                shot_splits=["shot_1_seed2", "shot_2_seed2", "shot_4_seed2", "shot_8_seed2", "shot_16_seed2"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["required_base_sections_found"], ["train", "val", "test"])
            self.assertEqual(
                report["required_support_splits_found"],
                ["shot_1_seed2", "shot_2_seed2", "shot_4_seed2", "shot_8_seed2", "shot_16_seed2"],
            )
            self.assertEqual(
                report["support_counts_by_shot"],
                {
                    "shot_1_seed2": 10,
                    "shot_2_seed2": 20,
                    "shot_4_seed2": 40,
                    "shot_8_seed2": 80,
                    "shot_16_seed2": 160,
                },
            )
            self.assertEqual(report["num_entries_checked"], 8)
            self.assertEqual(report["total_images_checked"], 350)
            self.assertTrue(any("matching manifest entries" in warning for warning in report["warnings"]))
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_features"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["saves_logits"])
            self.assertFalse(report["is_paper_result"])
            self.assertFalse(report["eligible_for_paper_tables"])

    def test_consumer_preflight_matches_older_summary_by_split_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = write_generic_summary_and_cache(
                root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id=None,
                split_name=None,
                split_path="splits/eurosat/shot_1_seed2.json",
                split_section="support",
                image_count=2,
                num_classes=2,
                run_key="old_support_without_split_id",
                seed=None,
                shot=None,
            )
            for section, count in [("train", 8), ("val", 4), ("test", 4)]:
                write_generic_summary_and_cache(
                    root,
                    dataset="eurosat",
                    backbone="remoteclip_vit_b32",
                    split_id=None,
                    split_name=None,
                    split_path="splits/eurosat/base_split_seed2.json",
                    split_section=section,
                    image_count=count,
                    num_classes=2,
                    run_key=f"old_base_{section}",
                    seed=None,
                    shot=None,
                )
            entries = [
                {"summary_path": str(path)}
                for path in sorted((root / "generic_features").rglob("feature_extraction_summary.json"))
            ]
            self.assertIn({"summary_path": entry["summary_path"]}, entries)
            manifest_path = root / "manifest" / "feature_cache_manifest.json"
            safe_write_json(manifest_path, {"entries": entries})

            report_path, is_valid = run_feature_cache_consumer_preflight(
                manifest_path=manifest_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split="base_seed2",
                shot_splits=["shot_1_seed2"],
                output_dir=root / "preflight",
                execution_env="remote_server",
                run_mode="local_validation",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertEqual(report["required_support_splits_found"], ["shot_1_seed2"])

    def test_consumer_preflight_cli_writes_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = make_manifest(
                root,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                num_classes=2,
                base_split="base_seed1",
                shot_splits=["shot_1_seed1"],
            )
            output_dir = root / "preflight"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_feature_cache_consumer_preflight.py",
                    "--manifest",
                    str(manifest_path),
                    "--dataset",
                    "eurosat",
                    "--backbone",
                    "remoteclip_vit_b32",
                    "--base-split",
                    "base_seed1",
                    "--shot-splits",
                    "shot_1_seed1",
                    "--output-dir",
                    str(output_dir),
                    "--execution-env",
                    "remote_server",
                    "--run-mode",
                    "local_validation",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report_path = extract_path(completed.stdout)
            self.assertEqual(report_path, output_dir / "feature_cache_consumer_preflight_report.json")
            self.assertTrue(read_json(report_path)["is_valid"])


def make_manifest(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    num_classes: int,
    base_split: str,
    shot_splits: list[str],
) -> Path:
    entries = []
    for section, count in [("train", 12), ("val", 6), ("test", 6)]:
        entries.append(write_summary_and_cache(root, dataset, backbone, base_split, section, count, num_classes))
    for split_id in shot_splits:
        shot = int(split_id.split("_")[1])
        entries.append(write_summary_and_cache(root, dataset, backbone, split_id, "support", shot * num_classes, num_classes))
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    return manifest_path


def write_summary_and_cache(
    root: Path,
    dataset: str,
    backbone: str,
    split_id: str,
    section: str,
    image_count: int,
    num_classes: int,
) -> dict[str, str]:
    run_dir = root / "features" / backbone / dataset / split_id / section / "run"
    cache_path = run_dir / "feature_cache.pt"
    summary_path = run_dir / "feature_extraction_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "image_features": [[0.0 for _ in range(512)] for _ in range(image_count)],
                "image_labels": [idx % num_classes for idx in range(image_count)],
                "image_paths": [f"fake://{idx}.jpg" for idx in range(image_count)],
                "split_name": section,
                "class_to_idx": {f"class_{idx}": idx for idx in range(num_classes)},
                "backbone": backbone,
                "dataset": dataset,
                "feature_dim": 512,
                "normalize_features": True,
                "created_at": "2026-05-07T00:00:00+00:00",
                "source_script": "tests/test_feature_cache_consumer_preflight.py",
                "metadata": {"dataset": dataset, "backbone": backbone},
            },
            handle,
        )
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": f"splits/{dataset}/{split_id}.json",
            "split_section": section,
            "image_count": image_count,
            "feature_shape": [image_count, 512],
            "feature_cache_path": str(cache_path),
            "run_dir": str(run_dir),
            "git_commit": "abc123",
            "checkpoint_loaded": True,
            "checkpoint_load_mode": "direct_state_dict",
            "checkpoint_num_tensors": 302,
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "final_weights_loaded_from_checkpoint": True,
            "final_weight_source": "cli_override_checkpoint",
            "final_checkpoint_load_status": "loaded_strictly_matching_keys",
            "is_real_feature_extraction": True,
            "is_full_feature_extraction": section != "support",
            "is_limited_real_extraction": False,
            "is_paper_result": False,
            "is_paper_result_candidate": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "extracts_text_features": False,
            "saves_predictions": False,
            "saves_logits": False,
            "start_time": "2026-05-07T00:00:00+00:00",
            "end_time": "2026-05-07T00:01:00+00:00",
            "total_time_sec": 60.0,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(cache_path), "run_dir": str(run_dir)}


def write_generic_summary_and_cache(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    split_id: str | None,
    split_name: str | None,
    split_path: str,
    split_section: str,
    image_count: int,
    num_classes: int,
    run_key: str,
    seed: int | None,
    shot: int | None,
    base_split: str | None = None,
    start_time: str = "2026-05-13T00:00:00+00:00",
    end_time: str = "2026-05-13T00:01:00+00:00",
) -> dict[str, str]:
    run_dir = root / "generic_features" / run_key / "run"
    cache_path = run_dir / "feature_cache.pt"
    summary_path = run_dir / "feature_extraction_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "image_features": [[0.0 for _ in range(512)] for _ in range(image_count)],
                "image_labels": [idx % num_classes for idx in range(image_count)],
                "image_paths": [f"fake://{run_key}/{idx}.jpg" for idx in range(image_count)],
                "split_name": split_section,
                "class_to_idx": {f"class_{idx}": idx for idx in range(num_classes)},
                "backbone": backbone,
                "dataset": dataset,
                "feature_dim": 512,
                "normalize_features": True,
                "created_at": "2026-05-13T00:00:00+00:00",
                "source_script": "tests/test_feature_cache_consumer_preflight.py",
                "metadata": {"dataset": dataset, "backbone": backbone},
            },
            handle,
        )
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "shot": shot,
            "split": split_path,
            "split_id": split_id,
            "split_name": split_name,
            "base_split": base_split,
            "split_path": split_path,
            "split_section": split_section,
            "image_count": image_count,
            "num_samples": image_count,
            "feature_shape": [image_count, 512],
            "feature_cache_path": str(cache_path),
            "run_dir": str(run_dir),
            "git_commit": "abc123",
            "checkpoint_loaded": True,
            "checkpoint_load_mode": "direct_state_dict",
            "checkpoint_num_tensors": 302,
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "final_weights_loaded_from_checkpoint": True,
            "final_weight_source": "cli_override_checkpoint",
            "final_checkpoint_load_status": "loaded_strictly_matching_keys",
            "is_real_feature_extraction": True,
            "is_full_feature_extraction": split_section != "support",
            "is_limited_real_extraction": False,
            "is_paper_result": False,
            "is_paper_result_candidate": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "extracts_text_features": False,
            "saves_predictions": False,
            "saves_logits": False,
            "start_time": start_time,
            "end_time": end_time,
            "total_time_sec": 60.0,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(cache_path), "run_dir": str(run_dir)}


def extract_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("feature_cache_consumer_preflight_report_path="):
            return Path(line.split("=", 1)[1])
    raise AssertionError(f"Missing report path in stdout: {stdout}")


if __name__ == "__main__":
    unittest.main()
