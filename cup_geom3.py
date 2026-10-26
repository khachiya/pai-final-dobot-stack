import numpy as np, pybullet as p, re, os
from sim_env import DobotStackEnv, URDF_PATH

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
dae = os.path.join(os.path.dirname(URDF_PATH), "meshes", "dae", "magicianSuctionCup.dae")
txt = open(dae, errors="ignore").read()

verts = []
for src in re.finditer(r'<source id="([^"]*)">(.*?)</source>', txt, re.S):
    sid, body = src.groups()
    if "position" not in sid.lower():
        continue
    am = re.search(r'<array[^>]*count="(\d+)"[^>]*>(.*?)</array>', body, re.S)
    if am:
        vals = np.array(list(map(float, am.group(2).split())))
        if len(vals) % 3 == 0 and len(vals) > 0:
            verts.append(vals.reshape(-1, 3))
print("MARK sources_found:", len(verts), "n_verts:", sum(len(v) for v in verts))
V = np.vstack(verts)

vo = np.array([-0.237150678405, 0.0, -0.0585951072678])
# link_5 world pose (link index 5)
pos, orn = p.getLinkState(env.arm, 5, computeForwardKinematics=True)
R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
W = (R @ (V + vo).T).T + np.array(pos)
zmin = W[:, 2].min()
near = W[W[:, 2] < zmin + 0.005]
print(f"MARK cup_bottom_z={zmin:.4f}  xy_at_bottom: x=[{near[:,0].min():.3f},{near[:,0].max():.3f}] y=[{near[:,1].min():.3f},{near[:,1].max():.3f}]")
print(f"MARK TCP={np.round(tip,4)}  delta=cup_bottom-TCP_z={zmin - tip[2]:+.4f} m")
