#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("matmul ok", float(x.sum()))

import pybullet as p
cid = p.connect(p.DIRECT)
print("pybullet api", p.getAPIVersion())
p.setAdditionalSearchPath(__import__("pybullet_data", fromlist=["x"]).getDataPath())
p.loadURDF("plane.urdf")
import sim_env
env = sim_env.DobotStackEnv(seed=1)
task = env.reset(color="red")
img = env.render()
print("render", img.shape, img.dtype, "task:", task)
traj = sim_env.run_expert(env)
print("expert steps:", len(traj), "success:", env.success())

import lerobot; print("lerobot", getattr(lerobot, "__version__", "?"))
for path in ("lerobot.common.policies.act.modeling_act", "lerobot.policies.act.modeling_act"):
    try:
        importlib.import_module(path); print("ACT module OK:", path)
    except Exception as e:
        print("no:", path, type(e).__name__)
for path in ("lerobot.common.policies.smolvla.modeling_smolvla", "lerobot.policies.smolvla.modeling_smolvla"):
    try:
        importlib.import_module(path); print("SmolVLA module OK:", path)
    except Exception as e:
        print("no:", path, type(e).__name__)
EOF
