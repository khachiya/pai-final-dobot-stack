#!/usr/bin/env bash
# PAI final v4: TCP = visual cup bottom -> regen demos, retrain ACT, evaluate.
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate

echo "[pipeline] $(date) start"
mv outputs/train/act_dobot outputs/train/act_dobot_v3 2>/dev/null || true
rm -rf data/dobot_stack_sim

echo "[pipeline] $(date) gen_data 300 episodes (true-contact env)"
python gen_data.py --episodes 300 --repo-id local/dobot_stack_sim \
    --root /home/khach/pai-final/data/dobot_stack_sim --seed 42 || { echo GEN_FAIL; exit 1; }

echo "[pipeline] $(date) train ACT 20k steps"
bash train_act.sh > train_act_v4.log 2>&1 || { echo TRAIN_FAIL; exit 1; }

echo "[pipeline] $(date) eval 20 episodes"
python eval_rollout.py --policy-type act \
    --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model \
    --episodes 20 > eval_act_v4.log 2>&1 || { echo EVAL_FAIL; exit 1; }

echo "[pipeline] $(date) ALL DONE"
grep -a RESULT eval_act_v4.log | tail -1
