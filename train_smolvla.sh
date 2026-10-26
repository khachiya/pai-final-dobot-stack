#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --dataset.repo_id=local/dobot_stack_sim \
  --dataset.root=/home/khach/pai-final/data/dobot_stack_sim \
  --output_dir=/home/khach/pai-final/outputs/train/smolvla_dobot \
  --job_name=smolvla_dobot \
  --batch_size=4 \
  --steps=20000 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --save_checkpoint=true
