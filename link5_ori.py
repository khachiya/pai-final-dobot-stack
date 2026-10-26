import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
for tag, x, y, z in [("home", 0.20, 0.0, 0.13), ("press", bx, by, 0.04),
                      ("lift", bx, by, 0.10), ("side", 0.10, 0.12, 0.10)]:
    env._move_to(x, y, z)
    ls = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
    pos, orn = np.array(ls[0]), np.array(ls[1])
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    # link5 local axes expressed in world
    print(f"MARK {tag} com={np.round(pos, 4)} lx={np.round(R[:,0], 3)} "
          f"ly={np.round(R[:,1], 3)} lz={np.round(R[:,2], 3)}", flush=True)
