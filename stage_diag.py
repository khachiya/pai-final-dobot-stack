import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=5)
env.reset(color="red")
t = (0.20, 0.0, 0.13)

q = env._solve_ik(*t)
for j, qv in zip(env.movable, q):
    p.resetJointState(env.arm, j, qv)
print("after LAPACK reset:", np.round(env._tip_pose()[:3], 4),
      "err mm", round(float(np.linalg.norm(env._tip_pose()[:3] - t)) * 1000, 1))

ok, err = env._refine_dls(t)
print("after DLS:", np.round(env._tip_pose()[:3], 4), "ok", ok, "err mm", round(err * 1000, 1))

for k in range(6):
    p.stepSimulation()
    print(f"step{k}:", np.round(env._tip_pose()[:3], 4),
          "joints", np.round([s[0] for s in p.getJointStates(env.arm, env.movable)], 3))
