import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
env._move_to(bx, by, 0.10)
tip = env._tip_pose()[:3]
env._move_to(bx, by, 0.10 - 0.06)
env._suction(1)
print("MARK carrying:", env.carrying, "tip:", np.round(env._tip_pose()[:3], 4), flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
# dark backdrop behind the action
v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.01, 0.5], rgbaColor=[0.02, 0.02, 0.03, 1])
p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.01, 0.5]),
                  v, [float(bpos[0]), float(bpos[1]) - 0.35, 0.25])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/dark_bg.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK darkbg done", flush=True)
