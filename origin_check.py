import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
env._move_to(0.15, 0.0, 0.15)
tip_rep = env._tip_pose()[:3]
ls = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
origin = np.array(ls[4])
com = np.array(ls[0])
print("MARK tip_reported(FK):", np.round(tip_rep, 4), flush=True)
print("MARK link5 origin:", np.round(origin, 4), flush=True)
print("MARK link5 COM:", np.round(com, 4), flush=True)
print("MARK FK - origin:", np.round(tip_rep[:3] - origin, 4), flush=True)
