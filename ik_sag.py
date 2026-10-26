import numpy as np, pybullet as p
from sim_env import DobotStackEnv, CUP_LOCAL

env = DobotStackEnv(seed=11)
env.reset(color="red")
env._move_to(0.20, 0.10, 0.18)
T = np.array([0.15, 0.0, 0.06])
ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
tgt_o = (T - R @ CUP_LOCAL).tolist()
q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o, [0, 0, 0, 1],
                                 lowerLimits=env.lo, upperLimits=env.hi,
                                 restPoses=env._warm, maxNumIterations=500)
for j, qv in zip(env.movable, q):
    p.resetJointState(env.arm, j, qv)
tip0 = env._tip_pose()[:3]
print(f"MARK right-after-IK err={np.linalg.norm(tip0-T)*1000:.1f}mm", flush=True)
for _ in range(2):
    p.stepSimulation()
tip1 = env._tip_pose()[:3]
print(f"MARK after-2-steps err={np.linalg.norm(tip1-T)*1000:.1f}mm", flush=True)
for _ in range(10):
    p.stepSimulation()
tip2 = env._tip_pose()[:3]
print(f"MARK after-12-steps err={np.linalg.norm(tip2-T)*1000:.1f}mm", flush=True)
