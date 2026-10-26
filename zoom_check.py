import numpy as np, pybullet as p, cv2
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=11)
env.reset(color="red")
bx = (env._spawn_xy[0]); by = env._spawn_xy[1]
size = env.box_size
btop = 0.0 + size

def step_to(x, y, z, suc):
    from sim_env import np as _np
    cur = env._tip_pose()
    a = np.array([x - cur[0], y - cur[1], z - cur[2], 0.0, float(suc)])
    env.step(a)

env._move_to(bx, by, btop + 0.05)
env._move_to(bx, by, btop - 0.004)
env._suction(1)
print("MARK carrying:", env.carrying)
step_to(bx, by, btop + 0.06, 1)   # lift with box
img = env.render(dist=0.22, target=[bx, by, btop + 0.03])
cv2.imwrite("/home/khach/pai-final/video/zoom_carry.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
h, w, _ = img.shape
col = img[:, w // 2, :].astype(int)
dark = (col.max(axis=1) < 80)
red = (col[:, 0] > 150) & (col[:, 1] < 100) & (col[:, 2] < 100)
ys_dark = np.where(dark)[0]
ys_red = np.where(red)[0]
# cup bottom = lowest dark pixel above the box region; box top = highest red pixel
box_top_y = ys_red.min() if len(ys_red) else -1
cup_above = ys_dark[ys_dark < box_top_y] if box_top_y > 0 else ys_dark
cup_bot_y = cup_above.max() if len(cup_above) else -1
print(f"MARK img_h={h} cup_bottom_row={cup_bot_y} box_top_row={box_top_y} gap_px={box_top_y - cup_bot_y}")
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/zoom_carry.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
