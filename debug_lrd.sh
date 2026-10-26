#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import importlib
for path in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
    try:
        m = importlib.import_module(path)
        print("OK:", path, hasattr(m, "LeRobotDataset"))
    except Exception as e:
        print("no:", path, type(e).__name__)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import inspect
sig = inspect.signature(LeRobotDataset.create)
print("create sig:", list(sig.parameters)[:10])
EOF
