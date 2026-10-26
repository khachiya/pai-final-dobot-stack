#!/usr/bin/env bash
# PAI final: regenerate demos with the real-model env, retrain ACT, evaluate.
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate

echo "[pipeline] $(date) start"
mv outputs/train/act_dobot outputs/train/act_dobot_v1_boxy 2>/dev/null || true
rm -rf data/dobot_stack_sim

echo "[pipeline] $(date) gen_data 300 episodes"
python gen_data.py --episodes 300 --repo-id local/dobot_stack_sim \
    --root /home/khach/pai-final/data/dobot_stack_sim --seed 42 || { echo GEN_FAIL; exit 1; }

echo "[pipeline] $(date) train ACT 20k steps"
bash train_act.sh > train_act_v2.log 2>&1 || { echo TRAIN_FAIL; exit 1; }

echo "[pipeline] $(date) eval 20 episodes"
python eval_rollout.py --policy-type act \
    --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model \
    --episodes 20 > eval_act_v2.log 2>&1 || { echo EVAL_FAIL; exit 1; }

echo "[pipeline] $(date) ALL DONE"
tail -3 eval_act_v2.log
