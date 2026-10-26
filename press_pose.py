import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
# press pose with CURRENT code (tip ~ btop-4mm old-TCP); render side closeup
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.004)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/press_pose.png",
            cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK press pose rendered")
