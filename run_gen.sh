#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
rm -rf data/dobot_stack_sim
python gen_data.py --episodes 300 --repo-id local/dobot_stack_sim \
    --root /home/khach/pai-final/data/dobot_stack_sim --seed 42
