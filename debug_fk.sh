#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
python - <<'EOF'
import numpy as np, pybullet as p
import sim_env
from sim_env import DobotStackEnv, L1, L2, H_BASE, GRIP_LEN

env = DobotStackEnv(seed=3)
print("joints:", [p.getJointInfo(env.arm, j)[1:3] for j in range(p.getNumJoints(env.arm))])
q0 = env._ik(0.2, 0.0, 0.15)
print("IK(home):", np.round(q0, 4))
for j in range(4): p.resetJointState(env.arm, j, q0[j])
pos, orn = p.getLinkState(env.arm, env.TOOL_LINK, computeForwardKinematics=True)[:2]
print("tool link world pos:", np.round(pos, 4), "euler:", np.round(p.getEulerFromQuaternion(orn), 3))
print("_tip_pose:", np.round(env._tip_pose(), 4))

# analytic FK from q0 to compare
j1, j2, j3 = q0[:3]
ex = L1*np.cos(j2); ez = H_BASE - L1*np.sin(j2)
wx = ex + L2*np.cos(j2+j3); wz = ez - L2*np.sin(j2+j3)
print("analytic wrist:", np.round([wx*np.cos(j1), wx*np.sin(j1), wz], 4))

env.reset(color="red")
for j in range(4):
    print(f"after reset q{j+1}={p.getJointState(env.arm, j)[0]:.4f}")
print("tip after reset _move_to:", np.round(env._tip_pose(), 4))
EOF
