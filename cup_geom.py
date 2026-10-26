import numpy as np, pybullet as p
from sim_env import DobotStackEnv

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
print("TCP:", np.round(tip, 4))

vsd = p.getVisualShapeData(env.arm)
for row in vsd:
    body, link, visIndex, geomType, dims, ws, orn, rgba, tex = row
    if link not in (4, 5, 6):
        continue
    print(f"link={link} geomType={geomType} dims={np.round(dims,4)} worldXYZ={np.round(ws,4)}")

for row in vsd:
    body, link, visIndex, geomType, dims, ws, orn, rgba, tex = row
    if link == 6:
        hz = dims[2] / 2.0
        print(f"EE visual center z={ws[2]:.4f} halfZ={hz:.4f} -> bottom z~{ws[2]-hz:.4f}")
        print(f"cup_bottom - TCP_z ~ {ws[2]-hz - tip[2]:+.4f} m")
