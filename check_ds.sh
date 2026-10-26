#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import json
d = json.load(open("data/dobot_stack_sim/meta/info.json"))
print(d["total_episodes"], "episodes,", d["total_frames"], "frames, fps", d["fps"])
tasks = open("data/dobot_stack_sim/meta/tasks.parquet", "rb")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/dobot_stack_sim", root="data/dobot_stack_sim")
print("dataset len:", len(ds))
s = ds[0]
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items()})
EOF
