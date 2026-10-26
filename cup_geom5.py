import numpy as np, pybullet as p, re, os
from sim_env import DobotStackEnv, URDF_PATH

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
dae = os.path.join(os.path.dirname(URDF_PATH), "meshes", "dae", "magicianSuctionCup.dae")
txt = open(dae, errors="ignore").read()

unit = re.search(r'<unit[^>]*meter="([^"]+)"', txt)
unit = float(unit.group(1)) if unit else 1.0
m = re.search(r'-mesh-positions-array"[^>]*>([\s\S]*?)</float_array>', txt)
V = np.array(list(map(float, m.group(1).split()))).reshape(-1, 3) * unit
print(f"MARK n_verts={len(V)} unit={unit} local_z=[{V[:,2].min():.4f},{V[:,2].max():.4f}]")

vo = np.array([-0.237150678405, 0.0, -0.0585951072678])
pos, orn = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
W = (R @ (V + vo).T).T + np.array(pos)
zmin = W[:, 2].min()
near = W[W[:, 2] < zmin + 0.004]
print(f"MARK cup_bottom_z={zmin:.4f} xy_at_bottom x=[{near[:,0].min():.3f},{near[:,0].max():.3f}] y=[{near[:,1].min():.3f},{near[:,1].max():.3f}]")
print(f"MARK TCP={np.round(tip,4)} delta=cup_bottom-TCP_z={zmin - tip[2]:+.4f} m")
