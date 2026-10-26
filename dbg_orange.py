import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
env._move_to(bx, by, 0.08)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
print("MARK box:", np.round(bpos, 4))
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + env.box_size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb)[:, :, :3].astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/dbg2.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
red = (R > 120) & (G < 80) & (B < 80)
print("MARK n_orange:", int(orange.sum()), "n_red:", int(red.sum()))
print("MARK img mean:", img.mean(axis=(0, 1)).round(1).tolist())
