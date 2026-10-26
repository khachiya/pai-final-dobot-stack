import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

for paint in [False, True]:
    env = DobotStackEnv(seed=11)
    env.reset(color="red")
    if paint:
        p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
    env._move_to(0.15, 0.0, 0.15)
    tip = env._tip_pose()[:3]
    tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    tag = "paint" if paint else "noplain"
    cv2.imwrite(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/cmp_{tag}.png",
                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
    dark = (R + G + B) < 200
    ys_o = np.where(orange.any(axis=1))[0]
    ys_d = np.where(dark.any(axis=1))[0]
    print(f"MARK {tag}: norange={int(orange.sum())} "
          f"orange_y={(int(ys_o.min()), int(ys_o.max())) if len(ys_o) else None} "
          f"dark_y={(int(ys_d.min()), int(ys_d.max())) if len(ys_d) else None}", flush=True)
print("MARK done", flush=True)
