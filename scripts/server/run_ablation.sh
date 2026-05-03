#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote GPU server.
# TODO_DATASET_ROOT: set in dataset configs or server split preparation.
# TODO_FEATURE_ROOT: directory containing precomputed real feature caches.
# TODO_CHECKPOINT_ROOT: directory containing local model checkpoints if required.
# TODO_RESULT_ROOT: server result root, with ablation raw metrics under results/raw.
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

for INIT in mean kmeans random_group_mean medoid; do
  python scripts/run_rs_cpc.py \
    --config configs/methods/rs_cpc.yaml \
    --env-config configs/env/remote_server.yaml \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --method rs_cpc \
    --shot "${SHOT}" \
    --seed "${SEED}" \
    --split "${SPLIT_PATH}" \
    --feature-cache "${FEATURE_ROOT}/${DATASET}/${BACKBONE}/feature_cache_seed${SEED}.pt" \
    --num-prototypes-per-class 4 \
    --prototype-init "${INIT}" \
    --fusion validation_tuned_alpha \
    --execution-env remote_server \
    --run-mode server_ablation \
    --device cuda \
    --output-dir "${RESULT_ROOT}/raw"
done
