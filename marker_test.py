import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.resetBasePositionAndOrientation(env.box, [9, 9, -1], [0, 0, 0, 1])
env._move_to(bx, by, 0.10)
tip = env._tip_pose()[:3]
print("MARK TCP:", np.round(tip, 4))

# bright green 6mm marker cubes along the cup axis at TCP+{0,10,20,30}mm
for dz in [0.0, 0.010, 0.020, 0.030]:
    s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.003] * 3)
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.003] * 3, rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, s, v, [tip[0], tip[1], tip[2] + dz])

tgt = [float(tip[0]), float(tip[1]), float(tip[2] + 0.015)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.18, 270, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
import cv2
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/markers.png",
            cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK markers done")
