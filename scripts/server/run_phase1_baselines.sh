#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote GPU server.
# TODO_DATASET_ROOT: set in dataset configs or pass to split/feature preparation commands on the server.
# TODO_FEATURE_ROOT: directory containing precomputed real feature caches.
# TODO_CHECKPOINT_ROOT: directory containing local model checkpoints if required.
# TODO_RESULT_ROOT: server result root, with raw metrics under results/raw.
# TODO_LOG_ROOT: server log root.

FEATURE_ROOT="TODO_FEATURE_ROOT"
CHECKPOINT_ROOT="TODO_CHECKPOINT_ROOT"
RESULT_ROOT="TODO_RESULT_ROOT"
LOG_ROOT="TODO_LOG_ROOT"
DATASET="nwpu_resisc45"
BACKBONE="remoteclip_vit_b32"
SHOT="16"
SEED="1"
SPLIT_PATH="splits/${DATASET}/shot_${SHOT}_seed${SEED}.json"

python scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --env-config configs/env/remote_server.yaml \
  --dataset "${DATASET}" \
  --backbone "${BACKBONE}" \
  --method zero_shot_clip \
  --shot "${SHOT}" \
  --seed "${SEED}" \
  --split "${SPLIT_PATH}" \
  --feature-cache "${FEATURE_ROOT}/${DATASET}/${BACKBONE}/feature_cache_seed${SEED}.pt" \
  --execution-env remote_server \
  --run-mode server_full \
  --device cuda \
  --output-dir "${RESULT_ROOT}/raw"

for METHOD in linear_probe tip_adapter proto_adapter; do
  python "scripts/run_${METHOD}.py" \
    --config "configs/methods/${METHOD}.yaml" \
    --env-config configs/env/remote_server.yaml \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --method "${METHOD}" \
    --shot "${SHOT}" \
    --seed "${SEED}" \
    --split "${SPLIT_PATH}" \
    --feature-cache "${FEATURE_ROOT}/${DATASET}/${BACKBONE}/feature_cache_seed${SEED}.pt" \
    --execution-env remote_server \
    --run-mode server_full \
    --device cuda \
    --output-dir "${RESULT_ROOT}/raw"
done
