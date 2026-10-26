import numpy as np, pybullet as p
from sim_env import DobotStackEnv


def Ry(a, v):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


env = DobotStackEnv(seed=11)
env.reset(color="red")
rng = np.random.default_rng(0)
A_rows, b_rows = [], []
for i in range(24):
    x = float(rng.uniform(0.08, 0.22))
    y = float(rng.uniform(-0.10, 0.10))
    z = float(rng.uniform(0.05, 0.20))
    env._move_to(x, y, z)
    q = [p.getJointState(env.arm, j)[0] for j in env.movable]
    ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
    o = np.array(ls[4])
    j1, j2, j3 = q[0], q[1], q[2]
    j4 = -(j2 + j3)
    # world = Rz(j1) @ Ry(j2) @ [Ry(j3) @ [Ry(j4) @ w + L1] + L2]
    M4 = np.array([[np.cos(j4), 0, np.sin(j4)], [0, 1, 0], [-np.sin(j4), 0, np.cos(j4)]])
    M3 = np.array([[np.cos(j3), 0, np.sin(j3)], [0, 1, 0], [-np.sin(j3), 0, np.cos(j3)]])
    M2 = np.array([[np.cos(j2), 0, np.sin(j2)], [0, 1, 0], [-np.sin(j2), 0, np.cos(j2)]])
    c, s = np.cos(j1), np.sin(j1)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # d = M2 M3 M4 w + M2 M3 L1 + M2 L2 ; world = Rz d
    B1 = Rz @ M2 @ M3 @ M4
    B2 = Rz @ M2 @ M3
    B3 = Rz @ M2
    Z = np.zeros((3, 3))
    A_rows.append(np.hstack([B1, B2, B3]))
    b_rows.append(o)
A = np.vstack(A_rows)
b = np.concatenate(b_rows)
par, res, *_ = np.linalg.lstsq(A, b, rcond=None)
pred = A @ par
err = np.abs(pred - b).reshape(-1, 3)
print("MARK max residual mm:", float(err.max() * 1000), flush=True)
print("MARK mean residual mm:", float(err.mean() * 1000), flush=True)
print("MARK w=", np.round(par[0:3], 4).tolist(), flush=True)
print("MARK L1=", np.round(par[3:6], 4).tolist(), flush=True)
print("MARK L2=", np.round(par[6:9], 4).tolist(), flush=True)
