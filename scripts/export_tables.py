#!/usr/bin/env python
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Result table exporter interface.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tables", nargs="+", default=["main"])
    parser.add_argument(
        "--include-run-modes",
        nargs="+",
        default=["server_full", "server_ablation", "server_benchmark"],
    )
    parser.add_argument(
        "--exclude-run-modes",
        nargs="+",
        default=["dry_run", "smoke_test", "debug", "tiny_subset", "local_validation"],
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit("Table export is reserved for a later phase and is not implemented in Phase 1A.")


if __name__ == "__main__":
    main()
