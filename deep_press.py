import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
# deep press: physical pose identical to future new-TCP press (cup at btop-2mm)
env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.030)
names = {i: p.getJointInfo(env.arm, i)[12].decode() for i in range(p.getNumJoints(env.arm))}
pts = p.getClosestPoints(env.arm, env.box, 0.10)
print("MARK closest pairs within 10cm (deep press):")
for c in pts:
    print(f"MARK linkA={c[3]}({names.get(c[3])}) dist={c[8]:+.4f}")

bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 270, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/deep_press.png",
            cv2.cvtColor(np.asarray(rgb)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
print("MARK deep press rendered")
