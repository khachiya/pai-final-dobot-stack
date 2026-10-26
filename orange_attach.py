import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv, run_expert

env = DobotStackEnv(seed=11)
p.changeVisualShape(env.arm, 5, rgbaColor=[1.0, 0.45, 0.0, 1.0])
env.reset(color="red")
traj = run_expert(env)
print("MARK expert ok:", env.success(), "frames:", len(traj), flush=True)
# rebuild attach moment exactly: fresh env, step to suction
env.reset(color="red")
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
def move(x, y, z):
    for _ in range(40):
        cur = env._tip_pose()
        d = np.array([x, y, z]) - cur[:3]
        if np.linalg.norm(d) < 0.002:
            break
        env.step(np.append(np.clip(d, -0.02, 0.02), [0.0, float(env.suction)]))
move(bx, by, btop + 0.05)
move(bx, by, btop + 0.001)
env._suction(1)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_attach.png",
            cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
img = np.asarray(rgb)[:, :, :3].astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
red = (R > 150) & (G < 70) & (B < 70)
ys_o = np.where(orange.any(axis=1))[0]
ys_r = np.where(red.any(axis=1))[0]
print("MARK orange rows:", int(ys_o.min()), int(ys_o.max()) if len(ys_o) else None, flush=True)
print("MARK red rows:", int(ys_r.min()), int(ys_r.max()) if len(ys_r) else None, flush=True)
