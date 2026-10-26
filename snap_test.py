import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size

def snap(tag):
    tip = env._tip_pose()[:3]
    ls = p.getLinkState(env.arm, 5)
    l5 = np.array(ls[0])
    bp, _ = p.getBasePositionAndOrientation(env.box)
    print(f"MARK {tag} tipz={tip[2]:.4f} link5z={l5[2]:.4f} boxz={bp[2]:.4f} "
          f"tipx={tip[0]:.3f} boxx={bp[0]:.3f} carry={int(env.carrying)}")

env._move_to(bx, by, btop + 0.05); snap("above")
env._move_to(bx, by, btop - 0.004); snap("press")
env._suction(1); snap("suction")
env._move_to(bx, by, btop + 0.06); snap("lift")
