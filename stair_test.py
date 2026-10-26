import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.resetBasePositionAndOrientation(env.box, [9, 9, -1], [0, 0, 0, 1])
env._move_to(bx, by, 0.12)
tip = env._tip_pose()[:3]
# stairstep plates: 2mm tall, 16mm wide, tops at tip+{22,24,26,28,30,32}mm, spread in x
for i, dz in enumerate([0.022, 0.024, 0.026, 0.028, 0.030, 0.032]):
    cx = tip[0] - 0.05 + i * 0.02
    s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.008, 0.02, 0.001])
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.008, 0.02, 0.001], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(0, s, v, [cx, tip[1], tip[2] + dz - 0.001])
tgt = [float(tip[0]), float(tip[1]), float(tip[2] + 0.027)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/stairs.png",
            cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK stairs done tip:", np.round(tip, 4))
