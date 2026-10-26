import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])

# fixed world ruler: green 4mm cubes at x=0.19, known world z
rz = [0.05 + i * 0.02 for i in range(9)]  # 0.05 .. 0.21
for z in rz:
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002] * 3, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002] * 3),
                      v, [0.19, 0.0, z])
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)

for tz in [0.08, 0.15, 0.20]:
    env._move_to(bx, 0.0, tz)
    tip = env._tip_pose()[:3]
    tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
    img = np.asarray(rgb).astype(int)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    green = (G > 120) & (R < 120) & (B < 120)
    ys = np.where(green.any(axis=1))[0]
    groups, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= 6:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)
    centers = sorted([float(np.mean(g)) for g in groups])
    # match detected centers (top->bottom) to visible ruler marks: use linear spacing to identify
    # ruler world z list; fit row->z using ALL marks assuming consecutive visible run; find offset by spacing
    if len(centers) >= 4:
        # spacing px per 20mm:
        sp = np.median(np.diff(centers))
        # anchor: brightest idea - fit with unknown integer offset via best alignment to rz grid
        best = None
        for k in range(len(rz)):
            # hypothesize centers[i] <-> rz[k + (i - (len(centers)-1))] ... simpler: try all alignments
            for start in range(len(rz) - len(centers) + 1):
                cand = rz[start:start + len(centers)][::-1]  # top row = highest z
                A = np.polyfit(centers, cand, 1)
                err = np.abs(np.polyval(A, centers) - cand).max()
                if best is None or err < best[0]:
                    best = (err, A, start)
        err, A, start = best
        dark = ((R + G + B) < 200)
        # cup x-range: darkest columns near center-left (cup was at ~276-282 before)
        per_col = dark[:, 200:320].sum(axis=0)
        cx = 200 + int(np.argmax(per_col))
        col = dark[:, cx - 3:cx + 4].any(axis=1)
        ys2 = np.where(col)[0]
        bottom = int(ys2.max()) if len(ys2) else -1
        zcup = A[0] * bottom + A[1]
        print(f"MARK tz={tz:.2f} tipz={tip[2]:.4f} fit_err={err*1000:.1f}mm cupbot_row={bottom} "
              f"cupZ={zcup:.4f} OFFSET={1000*(zcup-tip[2]):+.1f}mm", flush=True)
    else:
        print(f"MARK tz={tz:.2f} FEW markers: {len(centers)}", flush=True)
