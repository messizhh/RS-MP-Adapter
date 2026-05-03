#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote server after real jobs finish.
# TODO_DATASET_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_FEATURE_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_CHECKPOINT_ROOT: not used by table export, but must match the server experiment provenance.
# TODO_RESULT_ROOT: server result root containing raw metrics and table outputs.
# TODO_LOG_ROOT: server log root.

RESULT_ROOT="TODO_RESULT_ROOT"

python scripts/export_tables.py \
  --input-dir "${RESULT_ROOT}/raw" \
  --output-dir "${RESULT_ROOT}/tables" \
  --tables main efficiency cache_tradeoff per_class \
  --include-run-modes server_full server_ablation server_benchmark \
  --exclude-run-modes dry_run smoke_test debug tiny_subset local_validation
