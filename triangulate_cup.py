import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])

proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)

for tag, ax, ay, az in [("A", 0.15, 0.0, 0.15), ("B", 0.10, 0.05, 0.10)]:
    env._move_to(ax, ay, az)
    tip = env._tip_pose()[:3]
    ls = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
    origin = np.array(ls[4])
    R5 = np.array(p.getMatrixFromQuaternion(ls[1])).reshape(3, 3)
    ctr = []
    for yaw in [0, 90]:
        tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
        view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, yaw, -2, 0, 2)
        _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = np.asarray(rgb)[:, :, :3].astype(int)
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
        ys, xs = np.where(orange)
        if len(ys) < 50:
            print(f"MARK {tag} yaw{yaw}: NO ORANGE", flush=True)
            continue
        w = R[ys, xs].astype(float)
        cu, cv = float(np.sum(xs * w) / np.sum(w)), float(np.sum(ys * w) / np.sum(w))
        # also bottom edge (lowest orange row, median x there)
        yb = int(ys.max())
        xb = float(np.median(xs[ys >= yb - 3]))
        ctr.append((cu, cv, xb, yb, view))
        print(f"MARK {tag} yaw{yaw}: orange_n={len(ys)} centroid=({cu:.1f},{cv:.1f}) "
              f"bottom=({xb:.1f},{yb})", flush=True)
    if len(ctr) == 2:
        # triangulate bottom points
        pts3d = []
        for (cu, cv, xb, yb, view) in [ctr[0]]:
            pass
        # unproject both bottom observations, triangulate via least squares
        rays = []
        for (cu, cv, xb, yb, view) in ctr:
            V = np.array(view).reshape(4, 4, order="F")
            P = np.array(proj).reshape(4, 4, order="F")
            M = P @ V
            Minv = np.linalg.inv(M)
            for (uu, vv) in [(xb, yb)]:
                ndc = np.array([(uu - 256) / 256.0, (256 - vv) / 256.0, -1.0, 1.0])
                # near plane point; direction via camera center
                cam = np.linalg.inv(V) @ np.array([0, 0, 0, 1.0])
                wp = Minv @ ndc
                wp = wp[:3] / wp[3]
                d = wp - cam[:3]
                rays.append((cam[:3], d / np.linalg.norm(d)))
        # closest point between two rays
        (p1, d1), (p2, d2) = rays
        w0 = p1 - p2
        a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
        d, e = d1 @ w0, d2 @ w0
        t1 = (b * e - c * d) / (a * c - b * b)
        t2 = (a * e - b * d) / (a * c - b * b)
        cup = (p1 + t1 * d1 + p2 + t2 * d2) / 2
        cl = R5.T @ (cup - origin)
        print(f"MARK {tag} TRIANGULATED cup_world={np.round(cup,4).tolist()} "
              f"CUP_LOCAL={np.round(cl,4).tolist()}", flush=True)
