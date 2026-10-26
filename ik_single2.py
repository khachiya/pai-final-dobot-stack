import numpy as np, pybullet as p
from sim_env import DobotStackEnv, CUP_LOCAL

env = DobotStackEnv(seed=11)
env.reset(color="red")
# move far away first to test from non-home start
env._move_to(0.20, 0.10, 0.18)
for tag, T in [("A", [0.15, 0.0, 0.06]), ("B", [0.12, 0.08, 0.12])]:
    T = np.array(T)
    ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
    R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
    tgt_o = (T - R @ CUP_LOCAL).tolist()
    q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o, [0, 0, 0, 1],
                                     lowerLimits=env.lo, upperLimits=env.hi,
                                     restPoses=env._warm, maxNumIterations=500)
    for j, qv in zip(env.movable, q):
        p.resetJointState(env.arm, j, qv)
    tip = env._tip_pose()[:3]
    print(f"MARK {tag} single-call err={np.linalg.norm(tip-T)*1000:.1f}mm", flush=True)
