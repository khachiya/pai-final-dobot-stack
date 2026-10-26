import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])

for tz, tag in [(0.15, "high"), (0.06, "low")]:
    env._move_to(0.15, 0.0, tz)
    tip = env._tip_pose()[:3]
    tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    cv2.imwrite(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orx_{tag}.png",
                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
    ys, xs = np.where(orange)
    # weighted centroid (use R intensity as weight for robustness)
    w = R[ys, xs].astype(float)
    cx = float(np.sum(xs * w) / np.sum(w))
    cy = float(np.sum(ys * w) / np.sum(w))
    print(f"MARK {tag} tip={np.round(tip,4).tolist()} orange_n={len(ys)} "
          f"centroid_x={cx:.1f} (tip_px=256) dx_px={cx-256:+.1f} "
          f"bottom_row={int(ys.max())}", flush=True)
