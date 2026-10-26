import numpy as np, pybullet as p
from sim_env import DobotStackEnv, CUP_LOCAL

env = DobotStackEnv(seed=11)
env.reset(color="red")
T = np.array([0.15, 0.0, 0.06])
# single IK call from home, max iters
ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
tgt_o = (T - R @ CUP_LOCAL).tolist()
try:
    q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o, [0, 0, 0, 1],
                                     lowerLimits=env.lo, upperLimits=env.hi,
                                     restPoses=env._warm, maxNumIterations=500)
    print("MARK used maxNumIterations=500", flush=True)
except TypeError as e:
    print("MARK NO maxNumIterations kwarg:", e, flush=True)
    q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o, [0, 0, 0, 1],
                                     lowerLimits=env.lo, upperLimits=env.hi,
                                     restPoses=env._warm)
for j, qv in zip(env.movable, q):
    p.resetJointState(env.arm, j, qv)
tip = env._tip_pose()[:3]
print(f"MARK single-call tip={np.round(tip,4).tolist()} err={np.linalg.norm(tip-T)*1000:.1f}mm", flush=True)
