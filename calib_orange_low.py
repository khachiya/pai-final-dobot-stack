import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
env._move_to(bx, by, 0.06)  # low press-like pose (different joints from z=0.15)
tip = env._tip_pose()[:3]
print("MARK tip:", np.round(tip, 4), flush=True)
q = [p.getJointState(env.arm, j)[0] for j in env.movable]
print("MARK j2+j3+j4 =", f"{sum(q[1:4]):+.5f}", flush=True)

rz = [0.0 + i * 0.02 for i in range(8)]  # 0.00..0.14 world
for z in rz:
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002] * 3, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002] * 3),
                      v, [float(tip[0]) + 0.030, float(tip[1]), z])
tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb).astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_low.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
green = (G > 120) & (R < 120) & (B < 120)
ys, xs = np.where(orange)
print("MARK norange:", len(ys), "yrange:", (int(ys.min()), int(ys.max())) if len(ys) else None, flush=True)
gys = np.where(green.any(axis=1))[0]
groups, cur = [], [gys[0]]
for y in gys[1:]:
    if y - cur[-1] <= 6:
        cur.append(y)
    else:
        groups.append(cur)
        cur = [y]
groups.append(cur)
centers = sorted([float(np.mean(g)) for g in groups])
print("MARK green centers:", [int(c) for c in centers], flush=True)
if len(centers) >= 4 and len(ys) > 100:
    sp = float(np.median(np.diff(centers)))
    scale = 20.0 / sp
    ybot = int(ys.max())
    print(f"MARK scale={scale:.4f}mm/px cupbot_row={ybot} OFFSET={scale*(256-ybot):+.1f}mm", flush=True)
