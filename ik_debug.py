import numpy as np, pybullet as p
from sim_env import DobotStackEnv, CUP_LOCAL

env = DobotStackEnv(seed=11)
env.reset(color="red")
for tag, T in [("A", [0.15, 0.0, 0.06]), ("B", [0.12, 0.08, 0.12])]:
    T = np.array(T)
    for it in range(6):
        ls = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
        o = np.array(ls[4])
        R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        tgt_o = (T - R @ CUP_LOCAL).tolist()
        q = p.calculateInverseKinematics(env.arm, env.link_idx, tgt_o,
                                         lowerLimits=env.lo, upperLimits=env.hi,
                                         restPoses=env._warm)
        for j, qv in zip(env.movable, q):
            p.resetJointState(env.arm, j, qv)
        env._warm = list(q)
        tip = env._tip_pose()[:3]
        ls2 = p.getLinkState(env.arm, env.link_idx, computeForwardKinematics=True)
        R2 = np.array(p.getMatrixFromQuaternion(ls2[5])).reshape(3, 3)
        print(f"MARK {tag} it{it} tip={np.round(tip,4).tolist()} err={np.linalg.norm(tip-T)*1000:.1f}mm "
              f"lz={np.round(R2[:,2],3).tolist()}", flush=True)
