import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
env._move_to(bx, by, size + 0.05)
env._move_to(bx, by, size - 0.002)
env._suction(1)
q = [p.getJointState(env.arm, j)[0] for j in env.movable]
print("MARK q=", np.round(q, 4).tolist(), "j234sum=", round(float(sum(q[1:4])), 4), flush=True)
ls = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
R = np.array(p.getMatrixFromQuaternion(ls[1])).reshape(3, 3)
print("MARK link5 lz=", np.round(R[:, 2], 4).tolist(), flush=True)
print("MARK tip=", np.round(env._tip_pose()[:3], 4).tolist(), flush=True)
