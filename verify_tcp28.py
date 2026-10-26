import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.002)
env._suction(1)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/verify_tcp28.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # getCameraImage returns RGB (not BGR)
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
dark = (R + G + B) < 200
ys_r = np.where(red.any(axis=1))[0]
# dark structure bottom above the box: restrict to box x-range
xs_r = np.where(red.any(axis=0))[0]
cx0, cx1 = int(xs_r.min()), int(xs_r.max())
col = dark[:, cx0:cx1 + 1].any(axis=1)
ys_d = np.where(col)[0]
ys_d = ys_d[ys_d < ys_r.min()] if len(ys_r) and len(ys_d) else ys_d
print("MARK red_top_row:", int(ys_r.min()) if len(ys_r) else None,
      "dark_bottom_above:", int(ys_d.max()) if len(ys_d) else None, flush=True)
if len(ys_r) and len(ys_d):
    print(f"MARK gap_px={int(ys_r.min()) - int(ys_d.max())}", flush=True)
