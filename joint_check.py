import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
for z in [0.15, 0.08, 0.06, 0.10]:
    env._move_to(bx, by, z)
    q = [p.getJointState(env.arm, j)[0] for j in env.movable]
    print(f"MARK ztgt={z:.3f} j={[round(v, 4) for v in q]} j2+j3+j4={sum(q[1:4]):+.4f}", flush=True)
