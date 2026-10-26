import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
env._move_to(bx, by, 0.08)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
print("MARK box:", np.round(bpos, 4), "size:", env.box_size)
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + env.box_size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
ret = p.getCameraImage(256, 256, view, proj, renderer=p.ER_TINY_RENDERER)
print("MARK ret len:", len(ret), "rgb type:", type(ret[2]))
rgb = np.asarray(ret[2])
print("MARK rgb shape:", rgb.shape, "mean:", rgb.mean(axis=(0, 1)).round(1) if rgb.ndim == 3 else "N/A")
