"""Definitive visual cup-bottom measurement. No DAE math, no VLM, no color names.
Method: green markers at KNOWN world heights beside the cup -> pixel-to-world mapping.
Then locate lowest dark cup pixel on the tool axis -> true visual cup bottom in world Z.
"""
import numpy as np
import pybullet as p
import cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
env._move_to(bx, 0.0, 0.15)
tip = env._tip_pose()[:3]
print("MARK tip:", np.round(tip, 4), flush=True)
print("MARK TCP_OFF in use:", open("sim_env.py").read().split("TCP_OFF")[1].split("\n")[0].strip(), flush=True)

# green ruler markers 30mm to +x of tool axis, known world heights
heights = []
for dz in [-20, -10, 0, 10, 20, 30, 40, 50]:
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.002, 0.002], rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002] * 3),
                      v, [float(tip[0] + 0.030), float(tip[1]), float(tip[2] + dz / 1000.0)])
    heights.append(dz)

tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
img = np.asarray(rgb).astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_abs.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
green = (G > 120) & (R < 120) & (B < 120)
# group green blobs by row -> each marker's center row
ys = np.where(green.any(axis=1))[0]
groups, cur = [], [ys[0]]
for y in ys[1:]:
    if y - cur[-1] <= 6:
        cur.append(y)
    else:
        groups.append(cur)
        cur = [y]
groups.append(cur)
centers = sorted([int(np.mean(g)) for g in groups])
print("MARK green rows centers:", centers, flush=True)
# map: pixel row -> known height: fit using detected groups vs expected heights
# expected: markers sorted top-to-bottom = heights sorted desc (+50 at top)
if len(centers) >= 5:
    hs = sorted(heights, reverse=True)[:len(centers)]
    A = np.polyfit(centers, hs, 1)
    print(f"MARK px2world: h_mm = {A[0]:.4f}*row + {A[1]:.2f}", flush=True)
    # cup: darkest vertical structure near image center x (tool axis projects to center)
    col = img[:, 250:262, :].mean(axis=1)
    dark = col.sum(axis=1) < 200
    # longest dark run = cup+gripper column
    best, s, p0 = (0, 0, 0), None, None
    for y in range(512):
        if dark[y] and s is None:
            s = y
        if not dark[y] and s is not None:
            if y - s > best[1] - best[0]:
                best = (s, y, y - s)
            s = None
    if s is not None and 512 - s > best[1] - best[0]:
        best = (s, 512, 512 - s)
    print(f"MARK longest dark run: rows {best[0]}-{best[1]} (len {best[2]})", flush=True)
    cup_bottom_row = best[1]
    h_est = A[0] * cup_bottom_row + A[1]
    print(f"MARK cup bottom row={cup_bottom_row} -> world offset vs tip: {h_est:+.1f} mm", flush=True)
else:
    print("MARK ERROR: too few markers found:", len(centers), flush=True)
