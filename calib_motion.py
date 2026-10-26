import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
AX, AY = 0.10, 0.0  # away from box

rz = [0.05 + i * 0.02 for i in range(9)]
for z in rz:
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002] * 3, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002] * 3),
                      v, [0.19, 0.0, z])
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)


def render_at(tipz, fixed_view=None, fixed_tip=None):
    env._move_to(AX, AY, tipz)
    tip = env._tip_pose()[:3]
    if fixed_view is None:
        tgt = [float(tip[0]), float(tip[1]), float(tip[2])]
        view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    else:
        view = fixed_view
        tip = fixed_tip
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.asarray(rgb).astype(int), tip, view


for tz in [0.10, 0.16]:
    img1, tip1, view1 = render_at(tz)
    img2, _, _ = render_at(tz + 0.015, fixed_view=view1, fixed_tip=tip1)
    V = np.array(view1).reshape(4, 4, order="F")
    P = np.array(proj).reshape(4, 4, order="F")
    M = P @ V
    v = M @ np.array([tip1[0], tip1[1], tip1[2], 1.0])
    ndc = v[:3] / v[3]
    px, py = int((ndc[0] + 1) * 256), int((1 - ndc[1]) * 256)
    diff = np.abs(img1.astype(int) - img2.astype(int)).sum(axis=2)
    motion = diff > 60
    # cup column: tip projected x +-40 (cup is offset ~+20px in x from tip axis)
    x0, x1 = max(0, px - 40), min(512, px + 41)
    col = motion[:, x0:x1].any(axis=1)
    ys = np.where(col)[0]
    # exclude the ruler markers (they don't move - static, so diff cancels them - good)
    # lowest moving pixel on tip column = cup bottom (arm moved up 15mm, cup was lowest moving part)
    bot = int(ys.max()) if len(ys) else -1
    # ruler fit from img1
    R, G, B = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    green = (G > 120) & (R < 120) & (B < 120)
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
    best = None
    if len(centers) >= 4:
        for start in range(len(rz) - len(centers) + 1):
            cand = rz[start:start + len(centers)][::-1]
            A = np.polyfit(centers, cand, 1)
            err = np.abs(np.polyval(A, centers) - cand).max()
            if best is None or err < best[0]:
                best = (err, A)
        err, A = best
        zcup = A[0] * bot + A[1]
        print(f"MARK tz={tz:.2f} tip_proj=({px},{py}) cupbot_row={bot} cupZ={zcup:.4f} "
              f"tipZ={tip1[2]:.4f} OFFSET={1000*(zcup-tip1[2]):+.1f}mm fiterr={err*1000:.1f}mm",
              flush=True)
    else:
        print(f"MARK tz={tz:.2f} FEW markers", flush=True)
