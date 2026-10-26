import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.0, 1.0, 1.0])
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop + 0.002)
env._suction(1)
env._move_to(bx, by, btop + 0.06)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/mag2.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
col = img[:, 256, :]
out = []
for y in range(0, 512, 4):
    r, g, b = col[y]
    if r > 150 and g > 70 and g < 190 and b < 110:
        out.append("M")
    elif r > 100 and r > 2 * g and r > 2 * b:
        out.append("R")
    elif r + g + b < 120:
        out.append("K")
    elif r + g + b > 700:
        out.append(".")
    else:
        out.append("?")
print("MARK " + "".join(out), flush=True)
