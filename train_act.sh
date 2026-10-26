#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/dobot_stack_sim \
  --dataset.root=/home/khach/pai-final/data/dobot_stack_sim \
  --output_dir=/home/khach/pai-final/outputs/train/act_dobot \
  --job_name=act_dobot \
  --batch_size=8 \
  --steps=20000 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --save_checkpoint=true
