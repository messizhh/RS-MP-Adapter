#!/usr/bin/env bash
set -euo pipefail

# Template only. Table export should include server run modes by default.

python scripts/export_tables.py \
  --input-dir results/raw \
  --output-dir results/tables \
  --tables main efficiency cache_tradeoff ablation per_class \
  --include-run-modes server_full server_ablation server_benchmark \
  --exclude-run-modes dry_run smoke_test debug tiny_subset local_validation
