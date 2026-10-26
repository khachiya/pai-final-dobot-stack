import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size

def probe(Dmm):
    env._move_to(bx, by, btop + Dmm / 1000.0)
    tip = env._tip_pose()[:3]
    bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
    tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(256, 256, view, proj, renderer=p.ER_TINY_RENDERER)
    img = np.asarray(rgb)[:, :, :3].astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red = (R > 100) & (R > 2 * G) & (R > 2 * B)
    ys, xs = np.where(red)
    if len(ys) == 0:
        return float(tip[2]), None
    rtop, x0, x1 = ys.min(), xs.min(), xs.max()
    dark = (R + G + B) < 200
    below = dark[rtop:min(256, rtop + 12), x0:x1 + 1].sum()
    return tip[2], (rtop, below)

for D in [+30, +20, +10, +5, 0, -6, -12]:
    tipz, info = probe(D)
    print(f"MARK D={D:+d} tipz={tipz:.4f} redtop_below_darkpx={info}", flush=True)
print("MARK sweep done", flush=True)
