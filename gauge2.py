import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
size = env.box_size
env._move_to(bx, by, 0.14)
tip = env._tip_pose()[:3]
print("MARK tip:", np.round(tip, 4), flush=True)
# plates: tops at tip+{30,32,34,36,38}mm, alternating x offsets, 12mm wide, 2mm tall
heights = []
for i, dz in enumerate([30, 32, 34, 36, 38]):
    cx = tip[0] + (0.03 if i % 2 == 0 else -0.03)
    s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.006, 0.02, 0.001])
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.006, 0.02, 0.001], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(0, s, v, [cx, tip[1], tip[2] + dz / 1000.0 - 0.001])
    heights.append(dz)
tgt = [float(tip[0]), float(tip[1]), float(tip[2] + 0.034)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb)[:, :, :3].astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/gauge2.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
red = (R > 120) & (G < 80) & (B < 80)
# orange bottom per x-column near center; plate tops per x-column
for cx, tag in [(220, "L"), (256, "C"), (292, "R")]:
    co = np.where(orange[:, cx])[0]
    cr = np.where(red[:, cx])[0]
    print(f"MARK col{cx}{tag} orange=[{co.min() if len(co) else None},{co.max() if len(co) else None}] "
          f"red_top={cr.min() if len(cr) else None}", flush=True)
print("MARK gauge done", flush=True)
