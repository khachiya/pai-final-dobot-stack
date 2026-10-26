"""Probe the real Dobot Magician URDF in PyBullet: scale, joints, IK reachability, renders."""
import numpy as np
import pybullet as p
import pybullet_data
import cv2

URDF = "/home/khach/pai-final/models/magician/dobot_magician.urdf"
OUT = "/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/"

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
arm = p.loadURDF(URDF, useFixedBase=True)

n = p.getNumJoints(arm)
print("num_joints:", n)
movable = []
for i in range(n):
    info = p.getJointInfo(arm, i)
    jtype = info[2]
    axis = np.array(info[14], dtype=float) if len(info) > 14 else np.zeros(3)
    print(f"  [{i}] {info[1].decode()} type={jtype} axis={np.round(axis, 3)} "
          f"lo={info[8]:.2f} hi={info[9]:.2f}")
    if jtype in (0, 3):
        movable.append(i)

ee_idx = None
for i in range(n):
    info = p.getJointInfo(arm, i)
    if info[1].decode() == "magician_joint_end_effector":
        ee_idx = i   # child link index of that joint == i
print("ee_idx:", ee_idx)

q0 = [0.0] * n
for j in range(n):
    p.resetJointState(arm, j, 0.0)
ls = p.getLinkState(arm, ee_idx, computeForwardKinematics=True)
pos0, orn0 = ls[0], ls[1]
M = np.array(p.getMatrixFromQuaternion(ls[7])).reshape(3, 3)   # columns = local axes in world
print("EE home pos:", np.round(pos0, 4))
print("EE home X,Y,Z axes (world):", np.round(M[:, 0], 2), np.round(M[:, 1], 2), np.round(M[:, 2], 2))

aabb_lo, aabb_hi = p.getAABB(arm, -1)
print("robot AABB lo/hi:", np.round(aabb_lo, 3), np.round(aabb_hi, 3))


def target_orn(cup_axis_sign):
    """Orientation where EE local axis (sign*Z) points to world -Z; X forward."""
    Z = np.array([0., 0., cup_axis_sign * -1.0])   # local +Z -> down if sign=+1
    X = np.array([1., 0., 0.]); X -= Z * (X @ Z)
    X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    Rm = np.column_stack([X, Y, Z])               # local->world
    return p.getQuaternionFromMatrix(Rm.tolist())


for k, name in enumerate("XYZ"):
    print(f"home local {name} . world-Z:", round(float(M[k] @ [0, 0, -1]), 3))

targets = [(0.20, 0.00, 0.15), (0.22, 0.08, 0.045), (0.13, -0.15, 0.06),
           (0.13, 0.15, 0.06), (0.18, -0.08, 0.045)]
lower = [p.getJointInfo(arm, j)[8] for j in movable]
upper = [p.getJointInfo(arm, j)[9] for j in movable]

best = None
for sign in (1, -1):
    errs = []
    qso_all = []
    for t in targets:
        qsol = p.getInverseKinematics(arm, ee_idx, t, target_orn(sign),
                                      lowerLimits=lower, upperLimits=upper,
                                      restPoses=[0.0] * len(movable), solver=1)
        for j, qv in zip(movable, qsol):
            p.resetJointState(arm, j, qv)
        ep = np.array(p.getLinkState(arm, ee_idx, computeForwardKinematics=True)[0])
        errs.append(np.linalg.norm(ep - np.array(t)) * 1000)
        qso_all.append(list(qsol))
    print(f"cup sign={sign}: pos errors mm:", np.round(errs, 1))
    if best is None or sum(errs) < sum(best[2]):
        best = (sign, qso_all, errs)

sign, qso_all, errs = best
print("chosen cup axis sign:", sign)


def render(fn):
    view = p.computeViewMatrixFromYawPitchRoll([0.16, 0, 0.05], 0.55, 30, -60, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
    cv2.imwrite(fn, cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))


for j in movable:
    p.resetJointState(arm, j, 0.0)
render(OUT + "mag_home.png")
for tag, idx in [("pick", 1), ("stackL", 2)]:
    for j, qv in zip(movable, qso_all[idx]):
        p.resetJointState(arm, j, qv)
    render(OUT + f"mag_{tag}.png")

p.disconnect()
print("done")
