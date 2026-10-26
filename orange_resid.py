import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
env._move_to(bx, by, size + 0.05)
env._move_to(bx, by, size - 0.002)
env._suction(1)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_fixed.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
ys_o = np.where(orange.any(axis=1))[0]
ys_r = np.where(red.any(axis=1))[0]
print("MARK orange rows:", int(ys_o.min()), int(ys_o.max()), flush=True)
print("MARK red top:", int(ys_r.min()), flush=True)
print(f"MARK residual gap px: {int(ys_r.min()) - int(ys_o.max())}", flush=True)
