import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
p.resetBasePositionAndOrientation(env.box, [9, 9, -1], [0, 0, 0, 1])
env._move_to(bx, by, 0.10)
tip = env._tip_pose()[:3]
print("MARK TCP:", np.round(tip, 4))

# green ruler dashes 15mm to the +x side, every 5mm from -10 to +50
for dz in range(-10, 51, 5):
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.002, 0.0025],
                            rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002] * 3),
                      v, [tip[0] + 0.030, tip[1], tip[2] + dz / 1000.0])
tgt = [float(tip[0]), float(tip[1]), float(tip[2] + 0.020)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.25, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_side.png",
            cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK ruler side done")
