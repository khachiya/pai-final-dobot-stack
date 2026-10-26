import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.004)   # press pose (old TCP)
names = {i: p.getJointInfo(env.arm, i)[12].decode() for i in range(p.getNumJoints(env.arm))}
pts = p.getClosestPoints(env.arm, env.box, 0.05)
print("MARK closest arm-box pairs within 5cm:")
for c in pts:
    # (contactFlag, bodyA, bodyB, linkA, linkB, posA, posB, normal, dist, ...)
    print(f"MARK linkA={c[3]}({names.get(c[3])}) dist={c[8]:+.4f} "
          f"posA={[round(v,4) for v in c[5]]} posB={[round(v,4) for v in c[6]]}")
