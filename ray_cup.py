import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
print("TCP:", np.round(tip, 4))
names = {}
for j in range(p.getNumJoints(env.arm)):
    names[j] = p.getJointInfo(env.arm, j)[12].decode()
print("links:", names)

start = np.array([tip[0], tip[1], 0.5])
end = np.array([tip[0], tip[1], -0.2])
lo = None
for dx in np.arange(-0.03, 0.031, 0.004):
    for dy in np.arange(-0.03, 0.031, 0.004):
        a = start + np.array([dx, dy, 0.0])
        b = end + np.array([dx, dy, 0.0])
        hits = p.rayTest(tuple(a), tuple(b), reportAllHits=True)
        for h in hits:
            body, link, frac, _pt, _n = h
            if body != env.arm:
                continue
            z = a[2] + frac * (b[2] - a[2])
            xyd = np.hypot(a[0] + frac * (b[0] - a[0]) - tip[0],
                           a[1] + frac * (b[1] - a[1]) - tip[1])
            if xyd < 0.025:  # near TCP axis -> cup region
                if lo is None or z < lo[0]:
                    lo = (z, link)
print("lowest arm surface near TCP axis:", None if lo is None else f"z={lo[0]:.4f} link={names.get(lo[1])}")
if lo:
    print(f"delta = cup_bottom - TCP_z = {lo[0] - tip[2]:+.4f} m")
