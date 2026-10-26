import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])

for tag, ax, ay, az in [("press", 0.15, 0.0, 0.06), ("mid", 0.12, 0.08, 0.12), ("high", 0.18, -0.06, 0.18)]:
    env._move_to(ax, ay, az)
    tip = env._tip_pose()[:3]
    ls = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
    R5 = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
    tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
    err = np.linalg.norm(tip - np.array([ax, ay, az]))
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
    ys, xs = np.where(orange)
    print(f"MARK {tag} tip_err={err*1000:.1f}mm lz={np.round(R5[:,2],3).tolist()} "
          f"orange_y={int(ys.min())}-{int(ys.max())} orange_xc={int(np.median(xs))}", flush=True)
