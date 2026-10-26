import numpy as np, pybullet as p, re, os
from sim_env import DobotStackEnv, URDF_PATH

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
dae = os.path.join(os.path.dirname(URDF_PATH), "meshes", "dae", "magicianSuctionCup.dae")
txt = open(dae, errors="ignore").read()

m = re.search(r'-mesh-positions-array"[^>]*>([\s\S]*?)</float_array>', txt)
V = np.array(list(map(float, m.group(1).split()))).reshape(-1, 3)
mm = re.search(r'<matrix sid="transform">([\s\S]*?)(?:</matrix>|<instance_geometry)', txt)
vals = [float(x) for x in mm.group(1).replace("\n", " ").split()]
M = np.array(vals[:16]).reshape(4, 4, order="F")   # COLLADA = column-major!
print("MARK M[:3,:3]=\n", np.round(M[:3, :3], 6))
print("MARK M trans =", np.round(M[:3, 3], 6))
Vm = (M[:3, :3] @ V.T).T + M[:3, 3]

vo = np.array([-0.237150678405, 0.0, -0.0585951072678])
ls = p.getLinkState(env.arm, 5)
pos, orn = ls[0], ls[1]
R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
W = (R @ (Vm + vo).T).T + np.array(pos)
print(f"MARK cup z range world: [{W[:,2].min():.4f}, {W[:,2].max():.4f}]")
print(f"MARK TCP={np.round(tip,4)} delta_fixed = {W[:,2].min() - tip[2]:+.4f} m")
