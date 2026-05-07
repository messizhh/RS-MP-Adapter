#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote GPU server.
# TODO_DATASET_ROOT: remote dataset root containing images referenced by split files.
# TODO_FEATURE_ROOT: output root for full real feature caches.
# TODO_CHECKPOINT_ROOT: directory containing local model checkpoints.
# TODO_RESULT_ROOT: server result root for summaries or later tables.
# TODO_LOG_ROOT: server log root.

DATASET="nwpu_resisc45"
BACKBONE="remoteclip_vit_b32"
BACKBONE_CONFIG="configs/backbones/remoteclip_vit_b32.yaml"
DATASET_CONFIG="configs/datasets/nwpu_resisc45.yaml"
DATASET_ROOT="TODO_DATASET_ROOT"
FEATURE_ROOT="TODO_FEATURE_ROOT"
CHECKPOINT_ROOT="TODO_CHECKPOINT_ROOT"
RESULT_ROOT="TODO_RESULT_ROOT"
LOG_ROOT="TODO_LOG_ROOT"
REMOTECLIP_WEIGHTS_FILE="TODO_REMOTECLIP_WEIGHTS_FILE.pt"
SEED="1"
SPLIT_SECTION="test"
SPLIT_PATH="splits/${DATASET}/base_split_seed${SEED}.json"

if [[ "${DATASET_ROOT}" == TODO_* || "${FEATURE_ROOT}" == TODO_* || "${CHECKPOINT_ROOT}" == TODO_* || "${RESULT_ROOT}" == TODO_* || "${LOG_ROOT}" == TODO_* || "${REMOTECLIP_WEIGHTS_FILE}" == TODO_* ]]; then
  echo "error=Fill DATASET_ROOT, FEATURE_ROOT, CHECKPOINT_ROOT, RESULT_ROOT, LOG_ROOT, and REMOTECLIP_WEIGHTS_FILE before running on the server." >&2
  exit 2
fi

mkdir -p "${FEATURE_ROOT}" "${RESULT_ROOT}/summaries/feature_extraction" "${LOG_ROOT}"

python scripts/extract_features.py \
  --config "${BACKBONE_CONFIG}" \
  --dataset-config "${DATASET_CONFIG}" \
  --dataset "${DATASET}" \
  --dataset-root "${DATASET_ROOT}" \
  --backbone "${BACKBONE}" \
  --weights-path "${CHECKPOINT_ROOT}/${REMOTECLIP_WEIGHTS_FILE}" \
  --split "${SPLIT_PATH}" \
  --split-section "${SPLIT_SECTION}" \
  --batch-size 64 \
  --device cuda \
  --execution-env remote_server \
  --run-mode server_full \
  --allow-real-extraction \
  --paper-result-candidate \
  --output-dir "${FEATURE_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/feature_extraction_${DATASET}_${BACKBONE}_${SPLIT_SECTION}_seed${SEED}.log"

echo "guarded_feature_extraction_template_completed=true"
