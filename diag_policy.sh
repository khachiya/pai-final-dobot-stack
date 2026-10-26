#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import torch, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

ds = LeRobotDataset("local/dobot_stack_sim", root="data/dobot_stack_sim")
pol = ACTPolicy.from_pretrained("outputs/train/act_dobot/checkpoints/last/pretrained_model")
dev = "cuda"; pol.to(dev).eval()

# sample a mid-episode frame (approach phase) and compare predicted vs dataset action
for idx in [5, 30, 60]:
    s = ds[idx]
    batch = {k: v.unsqueeze(0) for k, v in s.items() if torch.is_tensor(v)}
    batch = {k: v.to(dev) for k, v in batch.items()}
    with torch.no_grad():
        a = pol.select_action(batch).squeeze().cpu().numpy()
    print(f"idx{idx} pred={np.round(a,4)} gt={np.round(s['action'].numpy(),4)}")

# dataset action stats
acts = np.stack([ds[i]["action"].numpy() for i in range(0, len(ds), 7)])
print("dataset action mean:", np.round(acts.mean(0), 4))
print("dataset action std :", np.round(acts.std(0), 4))
EOF
