import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
p.resetBasePositionAndOrientation(env.box, [9, 9, -1], [0, 0, 0, 1])
env._move_to(bx, by, 0.15)
tip = env._tip_pose()[:3]
print("MARK tip:", np.round(tip, 4), flush=True)
tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.25, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/cup_iso.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
ys, xs = np.where(orange)
print(f"MARK orange yrange: {ys.min()}-{ys.max()} xrange: {xs.min()}-{xs.max()} n={len(ys)}", flush=True)
print("MARK tip projects to image center (256,256); scale=0.414mm/px", flush=True)
