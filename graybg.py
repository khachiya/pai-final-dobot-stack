import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
env._move_to(bx, by, size + 0.05)
env._move_to(bx, by, size - 0.002)
env._suction(1)
env._move_to(bx, by, size + 0.06)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
# big mid-gray backdrop behind everything
v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.01, 0.6], rgbaColor=[0.45, 0.45, 0.45, 1])
p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.01, 0.6]),
                  v, [float(bpos[0]), float(bpos[1]) - 0.5, 0.35])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.24, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/graybg.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
col = img[:, 256, :]
out = []
for y in range(0, 512, 4):
    r, g, b = col[y]
    s = int(r) + int(g) + int(b)
    if r > 150 and g < 90 and b < 90:
        out.append("R")
    elif s < 120:
        out.append("K")
    elif abs(int(r) - 128) < 45 and abs(int(g) - 128) < 45 and abs(int(b) - 128) < 45:
        out.append("G")
    elif s > 700:
        out.append("W")
    else:
        out.append("?")
print("MARK " + "".join(out), flush=True)
