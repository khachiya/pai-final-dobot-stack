import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])

OUT = []
for D in [0, 10, 20, 30]:
    env._move_to(bx, by, btop + D / 1000.0)
    tip = env._tip_pose()[:3]
    bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
    tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red = (R > 120) & (G < 80) & (B < 80)
    orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
    ys_r, xs_r = np.where(red)
    if len(ys_r) == 0:
        OUT.append(f"D={D:+d} no-red"); continue
    cx = int(np.median(xs_r))
    col_o = np.where(orange[:, cx])[0]
    col_r = np.where(red[:, cx])[0]
    rtop = col_r.min() if len(col_r) else -1
    # orange bottom: lowest orange run with len>=3 above rtop-2
    cand = col_o[col_o < rtop - 2] if rtop > 0 and len(col_o) else []
    obot = -1
    if len(cand):
        run_top = cand[0]; prev = cand[0]; best = (cand[0], cand[0])
        for y in cand[1:]:
            if y > prev + 3:
                if prev - run_top > best[1] - best[0]:
                    best = (run_top, prev)
                run_top = y
            prev = y
        if prev - run_top > best[1] - best[0]:
            best = (run_top, prev)
        obot = best[1]
    OUT.append(f"D={D:+d} tip_err={(tip[2]-(btop+D/1000.0))*1000:+.1f}mm cx={cx} "
               f"redtop={rtop} obot={obot} gap_px={rtop-obot if obot>=0 else None}")
    cv2.imwrite(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/calib_D{D}.png",
                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK " + " | ".join(OUT))
# scale from box: red height at box column away from cup
R2, G2, B2 = img[:, :, 2], img[:, :, 1], img[:, :, 0]
red2 = (R2 > 120) & (G2 < 80) & (B2 < 80)
col2 = np.where(red2[:, 120])[0]
print("MARK boxcol_red_span:", col2.min() if len(col2) else None, col2.max() if len(col2) else None,
      f"(size {size * 1000:.1f}mm)")
