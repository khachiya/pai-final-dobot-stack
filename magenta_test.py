import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
# paint the whole link-5 mesh magenta: unambiguous cup footprint
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.0, 1.0, 1.0])
# press tip to 2mm above box top, suction on (gate passes), then lift
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop + 0.002)
env._suction(1)
env._move_to(bx, by, btop + 0.06)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/magenta.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb).astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
mag = (R > 150) & (G < 120) & (B > 150)
red = (R > 150) & (G < 80) & (B < 80)
ys_m, xs_m = np.where(mag)
ys_r, xs_r = np.where(red)
print(f"MARK magenta y:[{ys_m.min() if len(ys_m) else None},{ys_m.max() if len(ys_m) else None}] x:[{xs_m.min() if len(xs_m) else None},{xs_m.max() if len(xs_m) else None}]", flush=True)
print(f"MARK red y:[{ys_r.min() if len(ys_r) else None},{ys_r.max() if len(ys_r) else None}] x:[{xs_r.min() if len(xs_r) else None},{xs_r.max() if len(xs_r) else None}]", flush=True)
