"""Probe real Dobot Magician URDF: IK reachability with tool-down (home) orientation + renders."""
import numpy as np
import pybullet as p
import cv2

URDF = "/home/khach/pai-final/models/magician/dobot_magician.urdf"
OUT = "/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/"

cid = p.connect(p.DIRECT)
arm = p.loadURDF(URDF, useFixedBase=True)
n = p.getNumJoints(arm)
movable = [i for i in range(n) if p.getJointInfo(arm, i)[2] in (0, 3)]
# IK target: magician_link_5 (last movable link); TCP = its origin + (0,0,-0.06)
link_idx = next(i for i in range(n)
                if p.getJointInfo(arm, i)[1].decode() == "magician_joint_5")
TCP_OFF = np.array([0.0, 0.0, -0.06])

for j in movable:
    p.resetJointState(arm, j, 0.0)


def ee_pos():
    ls = p.getLinkState(arm, link_idx, computeForwardKinematics=True)
    return np.array(ls[4]) + TCP_OFF


print("EE home:", np.round(ee_pos(), 4), flush=True)
lo = [p.getJointInfo(arm, j)[8] for j in movable]
hi = [p.getJointInfo(arm, j)[9] for j in movable]

# task-relevant targets (x, y, z of TCP), tool keeps home orientation (cup straight down)
targets = {
    "home":   (0.207, 0.0, 0.075),
    "pickA":  (0.22, 0.08, 0.045),
    "pickB":  (0.14, -0.07, 0.045),
    "stackL": (0.13, -0.15, 0.06),
    "stackR": (0.13, 0.15, 0.06),
    "high":   (0.20, 0.0, 0.20),
}
sols = {}
for name, t in targets.items():
    tgt = np.array(t) - TCP_OFF          # link-5 origin target
    warm = [0.3] * len(movable)
    for _ in range(4):                    # warm-started CCD passes
        for j, qv in zip(movable, warm):
            p.resetJointState(arm, j, qv)
        q = p.calculateInverseKinematics(arm, link_idx, list(tgt), [0, 0, 0, 1],
                                         lowerLimits=lo, upperLimits=hi,
                                         restPoses=warm)
        warm = list(q)
    for j, qv in zip(movable, q):
        p.resetJointState(arm, j, qv)
    err = np.linalg.norm(ee_pos() - np.array(t)) * 1000
    orn = p.getLinkState(arm, link_idx, computeForwardKinematics=True)[5]
    print(f"{name:7s} {t} -> err {err:5.1f} mm  q={np.round(q, 2)}", flush=True)
    sols[name] = (list(q), err)


def render(fn):
    view = p.computeViewMatrixFromYawPitchRoll([0.16, 0, 0.04], 0.55, 30, -60, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_TINY_RENDERER)
    cv2.imwrite(fn, cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))


for j in movable:
    p.resetJointState(arm, j, 0.0)
render(OUT + "mag_home.png")
for name in ("pickA", "stackL"):
    for j, qv in zip(movable, sols[name][0]):
        p.resetJointState(arm, j, qv)
    render(OUT + f"mag_{name}.png")

p.disconnect()
print("done")
