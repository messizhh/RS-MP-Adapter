#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extract_features import save_fake_features_for_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature extraction interface.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-env", default="local_wsl")
    parser.add_argument("--run-mode", default="smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("Full feature extraction is not implemented in Phase 1A. Use --dry-run for fake features.")
    path = save_fake_features_for_smoke(f"{args.output_dir}/fake_features.pt")
    print(path)


if __name__ == "__main__":
    main()
