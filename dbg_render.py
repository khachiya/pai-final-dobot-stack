import numpy as np, cv2, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")

# force both renderers on the same pose for comparison
target = [0.16, 0, 0]
view = p.computeViewMatrixFromYawPitchRoll(target, env.cam_dist, env.cam_yaw, env.cam_pitch, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)

try:
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/v2_hw.png",
                cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))
    print("HW ok")
except Exception as e:
    print("HW fail:", type(e).__name__, str(e)[:120])

_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/v2_tiny.png",
            cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))
print("tiny ok")

img = env.render()
print("env.render shape:", img.shape, img.dtype)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/v2_env.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
env.close()
