import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
print("MARK box_size:", round(size, 4))
btop = size
env._move_to(bx, by, btop + 0.030)   # TCP exactly 30mm above box top
tip = env._tip_pose()[:3]
print("MARK tip:", np.round(tip, 4), "want_z:", round(btop + 0.030, 4))

bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.20, 270, -3, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(256, 256, view, proj, renderer=p.ER_TINY_RENDERER)
img = np.asarray(rgb)[:, :, :3].astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/cal_30mm.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
red = (R > G + 30) & (R > B + 30)
dark = (R + G + B) < 150
ys, xs = np.where(red)
print("MARK red yrange:", ys.min(), ys.max(), "h_px:", ys.max() - ys.min() + 1)
cx = (xs.min() + xs.max()) // 2
col = dark[:, cx]
dys = np.where(col)[0]
print("MARK dark runs bottom above redtop:", [(dys[i]) for i in range(len(dys)) if i == 0 or dys[i] > dys[i-1] + 3][:1],
      "redtop:", ys.min())
# lowest dark pixel strictly above red top region
above = dys[dys < ys.min() - 2]
print("MARK lowest dark above box:", above.max() if len(above) else None,
      "gap_px:", ys.min() - (above.max() if len(above) else ys.min()))
