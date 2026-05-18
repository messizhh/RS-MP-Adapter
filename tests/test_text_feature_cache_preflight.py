from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.check_text_feature_cache_preflight import run_text_feature_cache_preflight
from src.utils.io import read_json, safe_write_json


class TextFeatureCachePreflightTest(unittest.TestCase):
    def test_missing_text_feature_cache_is_valid_but_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing")

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight missing cache",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertFalse(report["text_feature_cache_exists"])
            self.assertFalse(report["text_feature_cache_ready"])
            self.assertEqual(report["errors"], [])
            self.assertTrue(any("separate text feature extraction" in item for item in report["recommendations"]))
            self.assertEqual(report["class_names"], ["class_0", "class_1", "class_2"])
            self.assertEqual(report["num_classes"], 3)
            self.assertEqual(report["expected_feature_dim"], 512)
            self.assertFalse(report["loads_model"])
            self.assertFalse(report["extracts_text_features"])
            self.assertFalse(report["computes_logits"])
            self.assertFalse(report["computes_accuracy"])
            self.assertFalse(report["evaluates_model"])
            self.assertFalse(report["trains_model"])
            self.assertFalse(report["saves_predictions"])
            self.assertFalse(report["writes_results_raw"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_existing_text_feature_cache_with_correct_shape_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="valid")

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight valid cache",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["is_valid"])
            self.assertTrue(report["text_feature_cache_exists"])
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["text_feature_cache_inspection"]["text_feature_shape"], [3, 512])
            self.assertEqual(report["selected_text_feature_cache_path"], str(case["text_cache_path"]))
            self.assertEqual(report["proposed_text_feature_cache_path"], str(case["text_cache_path"]))
            self.assertFalse((root / "results" / "raw").exists())

    def test_existing_text_feature_cache_with_bad_shape_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="bad_shape")

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight bad shape",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(report["text_feature_cache_exists"])
            self.assertFalse(report["text_feature_cache_ready"])
            self.assertTrue(any("text_features shape" in error for error in report["errors"]))
            self.assertIsNone(report["selected_text_feature_cache_path"])
            self.assertEqual(report["text_feature_cache_candidates"][0]["text_feature_shape"], [2, 512])
            self.assertFalse(report["text_feature_cache_candidates"][0]["selectable"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_multiple_candidates_prefers_latest_real_non_fake_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing")
            text_dir = case["text_cache_path"].parent
            dry_path = text_dir / "20260512T140000" / "text_feature_cache.pt"
            real_path = text_dir / "20260512T140232" / "text_feature_cache.pt"
            write_text_cache(
                dry_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed1",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=True,
                uses_fake_text_features=True,
                created_at="2026-05-12T14:00:00+00:00",
            )
            write_text_cache(
                real_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed1",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=False,
                uses_fake_text_features=False,
                created_at="2026-05-12T14:02:32+00:00",
            )

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight multiple candidates",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(report["selected_text_feature_cache_path"], str(real_path))
            self.assertEqual(len(report["text_feature_cache_candidates"]), 2)
            self.assertEqual(report["text_feature_cache_candidates"][0]["path"], str(real_path))
            self.assertFalse(report["text_feature_cache_candidates"][0]["dry_run"])
            self.assertFalse(report["text_feature_cache_candidates"][0]["uses_fake_text_features"])
            self.assertEqual(report["text_feature_cache_candidates"][0]["selection_rank"], 1)
            self.assertEqual(report["text_feature_cache_candidates"][1]["path"], str(dry_path))

    def test_current_feature_layout_timestamped_text_cache_is_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing", image_layout="current")
            timestamped_path = case["text_cache_path"].parent / "20260518T042340" / "text_feature_cache.pt"
            write_text_cache(
                timestamped_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed1",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=True,
                uses_fake_text_features=True,
                created_at="2026-05-18T04:23:40+00:00",
            )

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight current layout timestamped cache",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(
                report["proposed_text_feature_cache_path"],
                str(root / "features" / "eurosat" / "remoteclip_vit_b32" / "text" / "text_feature_cache.pt"),
            )
            self.assertEqual(report["selected_text_feature_cache_path"], str(timestamped_path))
            self.assertTrue(report["text_feature_cache_candidates"][0]["dry_run"])
            self.assertTrue(report["text_feature_cache_candidates"][0]["uses_fake_text_features"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_current_feature_layout_can_discover_legacy_nested_extractor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing", image_layout="current")
            legacy_path = (
                root
                / "features"
                / "remoteclip_vit_b32"
                / "eurosat"
                / "base_seed1"
                / "eurosat"
                / "remoteclip_vit_b32"
                / "text"
                / "20260518T042340"
                / "text_feature_cache.pt"
            )
            write_text_cache(
                legacy_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed1",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=False,
                uses_fake_text_features=False,
                created_at="2026-05-18T04:23:40+00:00",
            )

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight legacy nested extractor cache",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(report["selected_text_feature_cache_path"], str(legacy_path))
            self.assertEqual(len(report["text_feature_cache_candidates"]), 1)
            self.assertFalse(report["text_feature_cache_candidates"][0]["dry_run"])
            self.assertFalse(report["text_feature_cache_candidates"][0]["uses_fake_text_features"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_mismatched_base_split_cache_is_not_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing", image_layout="current", split_id="base_seed3")
            mismatched_path = case["text_cache_path"].parent / "20260518T051438" / "text_feature_cache.pt"
            write_text_cache(
                mismatched_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed2",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=False,
                uses_fake_text_features=False,
                created_at="2026-05-18T05:14:38+00:00",
            )

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight mismatched base split cache",
            )

            report = read_json(report_path)
            self.assertFalse(is_valid)
            self.assertFalse(report["is_valid"])
            self.assertTrue(report["text_feature_cache_exists"])
            self.assertFalse(report["text_feature_cache_ready"])
            self.assertIsNone(report["selected_text_feature_cache_path"])
            self.assertEqual(len(report["text_feature_cache_candidates"]), 1)
            candidate = report["text_feature_cache_candidates"][0]
            self.assertEqual(candidate["path"], str(mismatched_path))
            self.assertFalse(candidate["selectable"])
            self.assertIn("base_split mismatch", candidate["selection_reason"])
            self.assertTrue(any("base_split mismatch" in error for error in candidate["errors"]))
            self.assertFalse((root / "results" / "raw").exists())

    def test_matching_base_split_cache_is_selected_over_mismatched_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing", image_layout="current", split_id="base_seed3")
            text_dir = case["text_cache_path"].parent
            matching_path = text_dir / "20260518T051438" / "text_feature_cache.pt"
            mismatched_path = text_dir / "20260518T061438" / "text_feature_cache.pt"
            write_text_cache(
                matching_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed3",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=False,
                uses_fake_text_features=False,
                created_at="2026-05-18T05:14:38+00:00",
            )
            write_text_cache(
                mismatched_path,
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                split_id="base_seed2",
                num_classes=3,
                text_rows=3,
                feature_dim=512,
                dry_run=False,
                uses_fake_text_features=False,
                created_at="2026-05-18T06:14:38+00:00",
            )

            report_path, is_valid = run_text_feature_cache_preflight(
                manifest_path=case["manifest_path"],
                dataset="eurosat",
                backbone="remoteclip_vit_b32",
                base_split=str(case["base_split_path"]),
                output_dir=root / "outputs" / "preflight" / "text_features",
                execution_env="local_wsl",
                run_mode="local_validation",
                command="pytest text preflight matching base split cache",
            )

            report = read_json(report_path)
            self.assertTrue(is_valid)
            self.assertTrue(report["text_feature_cache_exists"])
            self.assertTrue(report["text_feature_cache_ready"])
            self.assertEqual(report["selected_text_feature_cache_path"], str(matching_path))
            candidates = {row["path"]: row for row in report["text_feature_cache_candidates"]}
            self.assertTrue(candidates[str(matching_path)]["selectable"])
            self.assertFalse(candidates[str(mismatched_path)]["selectable"])
            self.assertIn("base_split mismatch", candidates[str(mismatched_path)]["selection_reason"])
            self.assertFalse((root / "results" / "raw").exists())

    def test_results_raw_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case = write_fake_case(root, text_cache_mode="missing")

            with self.assertRaisesRegex(ValueError, "results/raw"):
                run_text_feature_cache_preflight(
                    manifest_path=case["manifest_path"],
                    dataset="eurosat",
                    backbone="remoteclip_vit_b32",
                    base_split=str(case["base_split_path"]),
                    output_dir=root / "results" / "raw" / "text_features",
                    execution_env="local_wsl",
                    run_mode="local_validation",
                    command="pytest text preflight reject results raw",
                )


def write_fake_case(
    root: Path,
    *,
    text_cache_mode: str,
    image_layout: str = "legacy",
    split_id: str = "base_seed1",
    text_cache_split_id: str | None = None,
) -> dict[str, Path]:
    dataset = "eurosat"
    backbone = "remoteclip_vit_b32"
    num_classes = 3
    feature_dim = 512
    cache_split_id = text_cache_split_id or split_id
    base_split_path = write_base_split(root, dataset=dataset, split_id=split_id, num_classes=num_classes)
    entries = []
    for section, count in [("train", 12), ("val", 6), ("test", 6)]:
        entries.append(
            write_image_summary(
                root,
                dataset=dataset,
                backbone=backbone,
                split_id=split_id,
                split_path=base_split_path,
                section=section,
                image_count=count,
                feature_dim=feature_dim,
                image_layout=image_layout,
            )
        )
    if image_layout == "current":
        text_cache_path = root / "features" / dataset / backbone / "text" / "text_feature_cache.pt"
    else:
        text_cache_path = root / "features" / backbone / dataset / split_id / "text" / "text_feature_cache.pt"
    if text_cache_mode != "missing":
        rows = num_classes - 1 if text_cache_mode == "bad_shape" else num_classes
        write_text_cache(
            text_cache_path,
            dataset=dataset,
            backbone=backbone,
            split_id=cache_split_id,
            num_classes=num_classes,
            text_rows=rows,
            feature_dim=feature_dim,
        )
    manifest_path = root / "manifest" / "feature_cache_manifest.json"
    safe_write_json(manifest_path, {"entries": entries})
    return {
        "manifest_path": manifest_path,
        "base_split_path": base_split_path,
        "text_cache_path": text_cache_path,
    }


def write_base_split(root: Path, *, dataset: str, split_id: str, num_classes: int) -> Path:
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    data = {
        "dataset": dataset,
        "seed": 1,
        "shot": None,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "train": make_rows(12, num_classes),
        "val": make_rows(6, num_classes),
        "test": make_rows(6, num_classes),
        "support": [],
    }
    path = root / "splits" / dataset / f"{split_id}.json"
    safe_write_json(path, data)
    return path


def make_rows(count: int, num_classes: int) -> list[dict[str, object]]:
    return [
        {"class_name": f"class_{idx % num_classes}", "label": idx % num_classes, "path": f"class_{idx % num_classes}/{idx}.jpg"}
        for idx in range(count)
    ]


def write_image_summary(
    root: Path,
    *,
    dataset: str,
    backbone: str,
    split_id: str,
    split_path: Path,
    section: str,
    image_count: int,
    feature_dim: int,
    image_layout: str = "legacy",
) -> dict[str, str]:
    if image_layout == "current":
        run_dir = root / "features" / dataset / backbone / section / "run"
    else:
        run_dir = root / "features" / backbone / dataset / split_id / section / "run"
    summary_path = run_dir / "feature_extraction_summary.json"
    safe_write_json(
        summary_path,
        {
            "dataset": dataset,
            "backbone": backbone,
            "split_path": str(split_path),
            "split_section": section,
            "image_count": image_count,
            "feature_shape": [image_count, feature_dim],
            "feature_cache_path": str(run_dir / "feature_cache.pt"),
            "run_dir": str(run_dir),
            "is_paper_result": False,
            "eligible_for_paper_tables": False,
            "trains_model": False,
            "evaluates_model": False,
            "extracts_text_features": False,
            "saves_predictions": False,
            "saves_logits": False,
        },
    )
    return {"summary_path": str(summary_path), "feature_cache_path": str(run_dir / "feature_cache.pt"), "run_dir": str(run_dir)}


def write_text_cache(
    path: Path,
    *,
    dataset: str,
    backbone: str,
    split_id: str,
    num_classes: int,
    text_rows: int,
    feature_dim: int,
    dry_run: bool = False,
    uses_fake_text_features: bool = False,
    created_at: str = "2026-05-12T00:00:00+00:00",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_to_idx = {f"class_{idx}": idx for idx in range(num_classes)}
    with path.open("wb") as handle:
        pickle.dump(
            {
                "text_features": [[float(row) for _ in range(feature_dim)] for row in range(text_rows)],
                "class_names": [f"class_{idx}" for idx in range(num_classes)],
                "class_to_idx": class_to_idx,
                "prompt_templates": ["a satellite photo of {}."],
                "dataset": dataset,
                "backbone": backbone,
                "base_split": split_id,
                "feature_dim": feature_dim,
                "num_classes": num_classes,
                "normalize_features": True,
                "source_script": "tests/test_text_feature_cache_preflight.py",
                "created_at": created_at,
                "git_commit": "abc123",
                "execution_env": "local_wsl",
                "run_mode": "local_validation",
                "dry_run": dry_run,
                "uses_fake_text_features": uses_fake_text_features,
                "is_paper_result": False,
            },
            handle,
        )


if __name__ == "__main__":
    unittest.main()
