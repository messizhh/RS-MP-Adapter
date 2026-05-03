#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote server after real jobs finish.
# TODO_DATASET_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_FEATURE_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_WEIGHT_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_OUTPUT_ROOT: server output root containing raw metrics and table outputs.

OUTPUT_ROOT="TODO_OUTPUT_ROOT"

python scripts/export_tables.py \
  --input-dir "${OUTPUT_ROOT}/raw" \
  --output-dir "${OUTPUT_ROOT}/tables" \
  --tables main efficiency cache_tradeoff per_class \
  --include-run-modes server_full server_ablation server_benchmark \
  --exclude-run-modes dry_run smoke_test debug tiny_subset local_validation
