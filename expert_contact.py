import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv, run_expert

env = DobotStackEnv(seed=11)
env.reset(color="red")
traj = run_expert(env)
print("MARK expert success:", env.success(), "frames:", len(traj), flush=True)

# replay to attach moment: find first carrying step by re-running with prints
env.reset(color="red")
from sim_env import DobotStackEnv as E
import sim_env as S
# manual expert steps to attach: replicate run_expert until suction
bx, by = float(env._spawn_xy[0]), float(env._spawn_xy[1])
size = env.box_size
btop = size
def move(x, y, z):
    for _ in range(30):
        cur = env._tip_pose()
        d = np.array([x, y, z]) - cur[:3]
        if np.linalg.norm(d) < 0.003:
            break
        a = np.array([d[0], d[1], d[2], 0.0, float(env.suction)])
        env.step(a)
move(bx, by, btop + 0.05)
move(bx, by, btop - 0.004)
env._suction(1)
print("MARK carrying:", env.carrying, flush=True)
bpos = np.array(p.getBasePositionAndOrientation(env.box)[0])
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb)[:, :, :3].astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/expert_contact.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
red = (R > 120) & (G < 80) & (B < 80)
ys, xs = np.where(red)
cx = int(np.median(xs)) if len(xs) else 256
col = img[:, cx, :].mean(axis=1)
# find white gap just above red top
rtop = np.where(red[:, cx])[0]
rtop = int(rtop.min()) if len(rtop) else -1
gap = 0
y = rtop - 1
while y > 0 and col[y] > 200:
    gap += 1
    y -= 1
print(f"MARK redtop={rtop} whitegap_px={gap} scale~0.36mm/px => ~{gap * 0.36:.1f}mm", flush=True)
