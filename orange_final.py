import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
print("MARK final-start boxsize:", round(size, 4), flush=True)

for D in (0, 15, 30):
    env._move_to(bx, by, btop + D / 1000.0)
    tip = env._tip_pose()[:3]
    bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
    tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
    red = (R > 120) & (G < 80) & (B < 80)
    yo, xo = np.where(orange)
    yr, xr = np.where(red)
    obot = yo.max() if len(yo) else -1
    rt = yr.min() if len(yr) else -1
    # box height for scale (away from cup x)
    col = np.where(red[:, 60])[0]
    print(f"MARKFINAL D={D:+d} tiperr={(tip[2]-(btop+D/1000.0))*1000:+.1f}mm "
          f"orange_bot={obot} red_top={rt} gap_px={rt-obot if obot>=0 and rt>=0 else None} "
          f"boxh_px={col.max()-col.min() if len(col) else None}", flush=True)
    cv2.imwrite(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/final_D{D}.png",
                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARKFINAL done", flush=True)
