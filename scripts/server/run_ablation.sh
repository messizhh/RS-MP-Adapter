#!/usr/bin/env bash
set -euo pipefail

# Template only. Do not execute locally.
# The user must manually edit TODO placeholders and run this on the remote GPU server.
# TODO_DATASET_ROOT: set in dataset configs or server split preparation.
# TODO_FEATURE_ROOT: directory containing precomputed real feature caches.
# TODO_WEIGHT_ROOT: directory containing local CLIP/RemoteCLIP/GeoRSCLIP weights.
# TODO_OUTPUT_ROOT: server output root for ablation raw results.

FEATURE_ROOT="TODO_FEATURE_ROOT"
OUTPUT_ROOT="TODO_OUTPUT_ROOT"
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
    --output-dir "${OUTPUT_ROOT}/raw"
done
