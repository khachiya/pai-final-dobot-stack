import numpy as np, pybullet as p
from sim_env import DobotStackEnv, CUP_LOCAL

env = DobotStackEnv(seed=11)
env.reset(color="red")
T = np.array([0.15, 0.0, 0.06])
for tag, orn in [("fixed", [0, 0, 0, 1]), ("current", None)]:
    # reset arm to home-ish
    env._move_to(0.20, 0.0, 0.15)
    for it in range(8):
        ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
        R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        tgt_o = (T - R @ CUP_LOCAL).tolist()
        o = orn if orn else ls[5]
        q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o, o,
                                         lowerLimits=env.lo, upperLimits=env.hi,
                                         restPoses=env._warm)
        for j, qv in zip(env.movable, q):
            p.resetJointState(env.arm, j, qv)
        env._warm = list(q)
        tip = env._tip_pose()[:3]
        ls2 = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
        R2 = np.array(p.getMatrixFromQuaternion(ls2[5])).reshape(3, 3)
        print(f"MARK {tag} it{it} err={np.linalg.norm(tip-T)*1000:.1f}mm lz={np.round(R2[:,2],3).tolist()}",
              flush=True)
