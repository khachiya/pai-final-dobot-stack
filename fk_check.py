import numpy as np, pybullet as p
from sim_env import DobotStackEnv, TCP_OFF

env = DobotStackEnv.__new__(DobotStackEnv)   # no scene; pure FK test below via loaded arm
env2 = DobotStackEnv(seed=1)
arm = env2.arm
movable = env2.movable

# 1) FK vs pybullet ground truth (link5 visual frame + TCP_OFF)
rng = np.random.default_rng(0)
print("FK-vs-PB check:")
for trial in range(4):
    q = rng.uniform([-1.2, -0.6, -0.8, -1.5, 0], [1.2, 1.2, 1.2, 1.5, 0])
    q[3] = -(q[1] + q[2])
    for j, v in zip(movable, q):
        p.resetJointState(arm, j, float(v))
    ls = p.getLinkState(arm, env2.link_idx, computeForwardKinematics=True)
    pb_tcp = np.array(ls[0]) + TCP_OFF      # visual frame (identity origin on link5)
    fk = DobotStackEnv._fk(q)
    print(f"  q={np.round(q[:4],2)} FK={np.round(fk,3)} PB={np.round(pb_tcp,3)} "
          f"d={np.linalg.norm(fk-pb_tcp)*1000:.1f}mm")

# 2) workspace of constrained FK (j4 derived): sample j1=0 grid
print("workspace scan (j1=0):")
zs, rs = [], []
for j2 in np.linspace(-1.0, 2.79, 60):
    for j3 in np.linspace(-1.2, 1.75, 60):
        t = DobotStackEnv._fk([0, j2, j3, -(j2 + j3), 0])
        zs.append(t[2]); rs.append(np.hypot(t[0], t[1]))
zs, rs = np.array(zs), np.array(rs)
print(f"  z range [{zs.min():.3f},{zs.max():.3f}]  r range [{rs.min():.3f},{rs.max():.3f}]")

# can FK reach task points?
for name, t in [("home", (0.20, 0, 0.13)), ("press", (0.204, 0.049, 0.038)),
                ("stackL", (0.13, -0.15, 0.04))]:
    best = min(((np.linalg.norm(DobotStackEnv._fk([0, j2, j3, -(j2+j3), 0]) - np.array(t) *
                   np.array([1, 1, 1])), j2, j3)
               for j2 in np.linspace(-1.0, 2.79, 200)
               for j3 in np.linspace(-1.2, 1.75, 200)))
    # include yaw: rotate target into j1=0 frame instead
    r_t = np.hypot(t[0], t[1]); th = np.arctan2(t[1], t[0])
    best2 = min(((np.linalg.norm(DobotStackEnv._fk([th, j2, j3, -(j2+j3), 0]) - np.array(t)),
                  th, j2, j3)
                for j2 in np.linspace(-1.0, 2.79, 150)
                for j3 in np.linspace(-1.2, 1.75, 150)))
    print(f"  {name:6s} grid-best err={best[0]*1000:5.1f}mm (yaw=0) / {best2[0]*1000:5.1f}mm (with yaw)")
env2.close()
