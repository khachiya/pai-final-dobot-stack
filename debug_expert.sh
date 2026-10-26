#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import numpy as np, pybullet as p
import sim_env
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=3)
task = env.reset(color="red")
bx, by, bz = p.getBasePositionAndOrientation(env.box)[0]
print("box spawn:", np.round([bx, by, bz], 3))
tip = env._tip_pose()
print("home tip:", np.round(tip, 3), " (want", sim_env.HOME_XYZR[:3], ")")

traj = sim_env.run_expert(env)
for _ in range(60): p.stepSimulation()
bp = p.getBasePositionAndOrientation(env.box)[0]
print("final box:", np.round(bp, 3), "stack target:", sim_env.STACKS[env.side])
print("steps:", len(traj), "success:", env.success())

# quick multi-seed check
ok_n = 0
for seed in range(8):
    e = DobotStackEnv(seed=100 + seed)
    e.reset()
    sim_env.run_expert(e)
    for _ in range(40): p.stepSimulation()
    ok_n += int(e.success())
print(f"expert success rate: {ok_n}/8")
EOF
