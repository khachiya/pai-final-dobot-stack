"""Scene preview: real-model arm + table + box, sim_env-style camera; also tests EGL renderer."""
import numpy as np
import pybullet as p
import pybullet_data
import cv2

URDF = "/home/khach/pai-final/models/magician/dobot_magician.urdf"
OUT = "/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/"

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

arm = p.loadURDF(URDF, useFixedBase=True)
n = p.getNumJoints(arm)
movable = [i for i in range(n) if p.getJointInfo(arm, i)[2] in (0, 3)]
link_idx = next(i for i in range(n)
                if p.getJointInfo(arm, i)[1].decode() == "magician_joint_5")
TCP_OFF = np.array([0.0, 0.0, -0.06])

# red box on table like the task
half = 0.02
sc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=[0.85, 0.05, 0.05, 1])
box = p.createMultiBody(0.04, sc, vs, [0.20, -0.05, half])


def ee_pos():
    ls = p.getLinkState(arm, link_idx, computeForwardKinematics=True)
    return np.array(ls[4]) + TCP_OFF


lo = [p.getJointInfo(arm, j)[8] for j in movable]
hi = [p.getJointInfo(arm, j)[9] for j in movable]


def ik(t, warm):
    tgt = list(np.array(t) - TCP_OFF)
    for _ in range(4):
        for j, qv in zip(movable, warm):
            p.resetJointState(arm, j, qv)
        q = p.calculateInverseKinematics(arm, link_idx, tgt, [0, 0, 0, 1],
                                         lowerLimits=lo, upperLimits=hi,
                                         restPoses=warm)
        warm = list(q)
    return warm


warm = [0.3] * len(movable)

# pose 1: hovering above the box (approach)
q = ik((0.20, -0.05, 0.09), warm)
for j, qv in zip(movable, q):
    p.resetJointState(arm, j, qv)
print("hover err mm:", round(float(np.linalg.norm(ee_pos() - np.array([0.20, -0.05, 0.09])) * 1000), 1))

def render(fn, hw=False):
    view = p.computeViewMatrixFromYawPitchRoll([0.16, 0, 0], 0.55, 30, -60, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    r = p.ER_BULLET_HARDWARE_OPENGL if hw else p.ER_TINY_RENDERER
    _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=r)
    cv2.imwrite(fn, cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))


render(OUT + "scene_tiny.png")
try:
    render(OUT + "scene_hw.png", hw=True)
    print("hardware GL OK")
except Exception as e:
    print("hardware GL failed:", type(e).__name__, str(e)[:100])

# pose 2: suction touching box top
q = ik((0.20, -0.05, 0.041), q)
for j, qv in zip(movable, q):
    p.resetJointState(arm, j, qv)
print("touch err mm:", round(float(np.linalg.norm(ee_pos() - np.array([0.20, -0.05, 0.041])) * 1000), 1))
render(OUT + "scene_touch.png")

p.disconnect()
print("done")
