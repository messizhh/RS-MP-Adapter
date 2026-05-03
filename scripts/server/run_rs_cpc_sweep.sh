#!/usr/bin/env bash
set -euo pipefail

# Template only. Execute manually on the remote server in a later phase.

for m in 1 2 4 8; do
  python scripts/run_rs_cpc.py \
    --config configs/methods/rs_cpc.yaml \
    --dataset nwpu_resisc45 \
    --backbone remoteclip_vit_b32 \
    --shot 16 \
    --split splits/nwpu_resisc45/shot_16_seed1.json \
    --seed 1 \
    --num-prototypes-per-class "${m}" \
    --prototype-init kmeans \
    --fusion validation_tuned_alpha \
    --execution-env remote_server \
    --run-mode server_full \
    --device cuda \
    --output-dir results/raw
done
