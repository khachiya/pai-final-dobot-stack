import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
env._move_to(bx, by, size + 0.05)
env._move_to(bx, by, size - 0.004)
env._suction(1)
tip = env._tip_pose()[:3]
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
print("MARK tip:", np.round(tip, 4), "boxtop:", round(bpos[2] + size / 2, 4), flush=True)
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
V = np.array(view).reshape(4, 4, order="F")
P = np.array(proj).reshape(4, 4, order="F")
M = P @ V
cam = np.linalg.inv(V) @ np.array([0, 0, 0, 1.0])
print("MARK cam pos:", np.round(cam[:3], 4), flush=True)
for name, pt in [("tip", tip), ("boxtop", [bpos[0], bpos[1], bpos[2] + size / 2]),
                 ("boxbot", [bpos[0], bpos[1], bpos[2] - size / 2])]:
    v = M @ np.array([pt[0], pt[1], pt[2], 1.0])
    ndc = v[:3] / v[3]
    print(f"MARK proj {name}: px=({int((ndc[0]+1)*256)},{int((1-ndc[1])*256)})", flush=True)
