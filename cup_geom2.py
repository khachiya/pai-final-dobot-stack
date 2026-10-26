import numpy as np, pybullet as p, re, os
from sim_env import DobotStackEnv, URDF_PATH

env = DobotStackEnv(seed=1)
tip = env._tip_pose()[:3]
print("MARK TCP:", np.round(tip, 4))
xml = open(URDF_PATH).read()
base_dir = os.path.dirname(URDF_PATH)

joints = {}
for m in re.finditer(r'<joint[^>]*name="([^"]+)"[^>]*type="([^"]+)">(.*?)</joint>', xml, re.S):
    jn, jt, jb = m.groups()
    child = re.search(r'<child link="([^"]+)"', jb)
    parent = re.search(r'<parent link="([^"]+)"', jb)
    orig = re.search(r'<origin[^>]*xyz="([^"]*)"(?:[^>]*rpy="([^"]*)")?', jb)
    if child and parent:
        o = np.array(list(map(float, orig.group(1).split()))) if orig else np.zeros(3)
        r = np.array(list(map(float, orig.group(2).split()))) if orig and orig.group(2) else np.zeros(3)
        joints[child.group(1)] = (jn, jt, parent.group(1), o, r)

def link_pose_world(link):
    xyz = np.zeros(3); R = np.eye(3)
    chain = []
    cur = link
    while cur in joints:
        jn, jt, par, o, r = joints[cur]
        ang = 0.0
        if jt == "revolute":
            idx = p.getJointIndex(jn)
            if idx >= 0:
                ang = p.getJointState(env.arm, idx)[0]
        chain.append((o, r, ang))
        cur = par
    for o, r, ang in reversed(chain):
        R = R @ np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(r))).reshape(3, 3)
        xyz = xyz + R @ o
        Rz = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, ang]))).reshape(3, 3)
        R = R @ Rz
    return xyz, R

for m in re.finditer(r'<link name="([^"]+)">(.*?)</link>', xml, re.S):
    name, body = m.group(1), m.group(2)
    for v in re.finditer(r"<visual>(.*?)</visual>", body, re.S):
        vb = v.group(1)
        mesh = re.search(r'filename="([^"]+)"', vb)
        if not mesh:
            continue
        orig = re.search(r'<origin[^>]*xyz="([^"]*)"(?:[^>]*rpy="([^"]*)")?', vb)
        vo = np.array(list(map(float, orig.group(1).split()))) if orig else np.zeros(3)
        vr = np.array(list(map(float, orig.group(2).split()))) if orig and orig.group(2) else np.zeros(3)
        path = os.path.normpath(os.path.join(base_dir, mesh.group(1)))
        txt = open(path, errors="ignore").read()
        verts = []
        for am in re.finditer(r'<array[^>]*count="(\d+)"[^>]*>(.*?)</array>', txt, re.S):
            vals = np.array(list(map(float, am.group(2).split())))
            if len(vals) >= 3 and len(vals) % 3 == 0:
                verts.append(vals.reshape(-1, 3))
        if not verts:
            continue
        V = np.vstack(verts)
        Rv = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(vr))).reshape(3, 3)
        lxw, Rw = link_pose_world(name)
        W = (Rw @ ((Rv @ V.T).T + vo)).T + lxw
        print(f"MARK link={name} mesh={os.path.basename(path)} zmin={W[:,2].min():.4f} zmax={W[:,2].max():.4f}")
