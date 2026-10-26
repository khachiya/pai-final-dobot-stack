import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
env._move_to(bx, by, 0.10)
tip = env._tip_pose()[:3]
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb)
print("MARK shape:", img.shape, "dtype:", img.dtype, flush=True)
c0, c1, c2 = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
print("MARK ch0 mean/max:", round(c0.mean(), 1), c0.max(),
      "ch1:", round(c1.mean(), 1), c1.max(), "ch2:", round(c2.mean(), 1), c2.max(), flush=True)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/chan.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
# reddest pixel = box (known red material)
dom = c0.astype(int) - c1.astype(int)
ys, xs = np.where(dom > 80)
print("MARK reddest-dom pixels:", len(ys), "yrange:", (ys.min(), ys.max()) if len(ys) else None, flush=True)
