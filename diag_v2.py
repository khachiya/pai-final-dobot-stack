import numpy as np, pybullet as p
from sim_env import DobotStackEnv, TCP_OFF

env = DobotStackEnv(seed=5)
env.reset(color="red")


def tcp():
    return env._tip_pose()[:3]


bx, by, _ = p.getBasePositionAndOrientation(env.box)[0]
print("box at", np.round([bx, by], 3), "size", round(env.box_size, 3))

for name, t in [("home", (0.20, 0.0, 0.13)), ("above", (bx, by, env.box_size + 0.05)),
                ("press", (bx, by, env.box_size - 0.002))]:
    ok = env._move_to(*t)
    e = np.linalg.norm(tcp() - np.array(t)) * 1000
    print(f"{name:6s} target={np.round(t,3)} err={e:5.1f} mm ok={ok}", flush=True)

env._suction(1)
print("carrying after suction:", env.carrying)

# Jacobian row-order sanity check at current config
q = list(env._warm)
ls0 = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
t0 = np.array(ls0[4]) + TCP_OFF
dq = np.zeros(5); dq[1] = 0.02   # bump joint 2
for j, v in zip(env.movable, q + dq):
    p.resetJointState(env.arm, j, v)
ls1 = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
t1 = np.array(ls1[4]) + TCP_OFF
fd_lin = (t1 - t0) / 0.02
J = np.array(p.calculateJacobian(env.arm, env.link_idx, TCP_OFF.tolist(), q, [0.0]*5, [0.0]*5))
if J.ndim == 3:
    print("J nested shape:", J.shape)
    J = np.vstack([J[0], J[1]])
print("FD lin d/dq2 :", np.round(fd_lin, 3))
print("J row0..2 col1 (lin):", np.round(J[:3, 1], 3))
print("J row3..5 col1 (ang):", np.round(J[3:, 1], 3))
