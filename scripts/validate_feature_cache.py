#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_cache import load_feature_cache, shape_of_2d
from src.utils.io import safe_write_json
from src.utils.timing import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a saved feature cache without modifying it.")
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_feature_cache(args.feature_cache, args.execution_env, args.run_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = unique_path(output_dir / "feature_cache_validation.json")
    safe_write_json(report_path, report)
    print(f"validation_report_path={report_path}")
    print(f"is_valid={str(report['is_valid']).lower()}")
    if not report["is_valid"]:
        raise SystemExit(2)


def validate_feature_cache(path: str | Path, execution_env: str, run_mode: str) -> dict[str, object]:
    errors: list[str] = []
    cache = None
    try:
        cache = load_feature_cache(path)
        cache.validate()
    except Exception as exc:
        errors.append(str(exc))
    report = {
        "feature_cache_path": str(path),
        "execution_env": execution_env,
        "run_mode": run_mode,
        "is_paper_result": False,
        "is_valid": not errors,
        "errors": errors,
        "created_at": utc_now_iso(),
        "source_script": "scripts/validate_feature_cache.py",
    }
    if cache is not None:
        image_shape = shape_of_2d(cache.image_features)
        text_shape = shape_of_2d(cache.text_features) if cache.text_features is not None else ()
        report.update(
            {
                "dataset": cache.dataset,
                "backbone": cache.backbone,
                "num_images": image_shape[0],
                "feature_dim": cache.feature_dim,
                "num_classes": len(cache.class_to_idx),
                "num_text_features": text_shape[0] if text_shape else 0,
                "has_text_features": cache.text_features is not None,
                "uses_fake_data": bool(cache.metadata.get("uses_fake_data", False)),
                "uses_fake_features": bool(cache.metadata.get("uses_fake_features", False)),
            }
        )
    return report


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find non-existing output path for {path}")


if __name__ == "__main__":
    main()
