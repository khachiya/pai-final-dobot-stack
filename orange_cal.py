import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])

def measure(Dmm):
    env._move_to(bx, by, btop + Dmm / 1000.0)
    tip = env._tip_pose()[:3]
    bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
    tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(256, 256, view, proj, renderer=p.ER_TINY_RENDERER)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
    red = (R > 120) & (G < 80) & (B < 80)
    yo, xo = np.where(orange)
    yr, xr = np.where(red)
    if len(yr) == 0 or len(yo) == 0:
        print(f"MARK D={Dmm:+d}mm tip_err={(tip[2] - (btop + Dmm / 1000.0)) * 1000:+.1f}mm "
              f"n_orange={len(yo)} n_red={len(yr)} SKIP")
        return img
    # overlap: orange pixels inside red bbox
    inred = orange[yr.min():yr.max() + 1, xr.min():xr.max() + 1].sum()
    # white gap between orange bottom and red top (center column)
    cx = (xr.min() + xr.max()) // 2
    col_o = np.where(orange[:, cx])[0]
    rtop = np.where(red[:, cx])[0]
    rtop = rtop.min() if len(rtop) else -1
    below = col_o[col_o < rtop - 1] if rtop > 0 and len(col_o) else []
    gap = (rtop - below.max()) if len(below) else None
    print(f"MARK D={Dmm:+d}mm tip_err={(tip[2] - (btop + Dmm / 1000.0)) * 1000:+.1f}mm "
          f"orange_y=[{yo.min() if len(yo) else '-'},{yo.max() if len(yo) else '-'}] "
          f"red_y=[{yr.min() if len(yr) else '-'},{yr.max() if len(yr) else '-'}] "
          f"overlap_px={inred} gapcol_px={gap}")
    return img

for D in [+40, +30, +20, +10, 0, -10]:
    measure(D)
print("MARK binary done")
