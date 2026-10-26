import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.004)
env._suction(1)
env._move_to(bx, by, btop + 0.06)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
print("MARK carry:", env.carrying, "boxtop:", round(float(bpos[2] + size / 2), 4))
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
for yaw in (0, 90, 180, 270):
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.25, yaw, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    cv2.imwrite(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/q_yaw_{yaw}.png",
                cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK quad done")
