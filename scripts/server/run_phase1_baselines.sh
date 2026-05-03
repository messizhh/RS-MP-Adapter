#!/usr/bin/env bash
set -euo pipefail

# Template only. Execute manually on the remote server after dataset paths and splits are prepared.

python scripts/run_zero_shot.py \
  --config configs/methods/zero_shot_clip.yaml \
  --env-config configs/env/remote_server.yaml \
  --dataset nwpu_resisc45 \
  --backbone remoteclip_vit_b32 \
  --split splits/nwpu_resisc45/base_split_seed1.json \
  --seed 1 \
  --execution-env remote_server \
  --run-mode server_full \
  --device cuda \
  --output-dir results/raw
