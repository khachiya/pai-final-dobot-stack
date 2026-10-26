"""Camera/pose contact sheet for the real-model env; checks shadow flag."""
import numpy as np, cv2, pybullet as p
import pybullet_data
from sim_env import DobotStackEnv, run_expert

OUT = "/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/"

env = DobotStackEnv(seed=3)
print("shadow attr exists:", hasattr(p, "COFIGURE_DEBUG_VISUALIZER_SHADOW_ENABLED"))
try:
    print("shadow flag now:", p.readDebugVisualizerSettings(physicsClientId=env.cid).get(
        "COFIGURE_DEBUG_VISUALIZER_SHADOW_ENABLED", "?"))
except Exception as e:
    print("read vis settings failed:", type(e).__name__)

env.reset(color="red")
# park arm at hover over box for a representative pose set
import math
bx, by, _ = p.getBasePositionAndOrientation(env.box)[0]
env._move_to(bx, by, 0.09, iters=60)


def shot(target, dist, yaw, pitch, w=512):
    view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(w, w, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return rgb[:, :, :3].astype(np.uint8)


cams = {
    "A_current": ([0.16, 0, 0], 0.55, 30, -60),
    "B_side":    ([0.14, 0, 0.07], 0.62, 90, -28),
    "C_oblique": ([0.14, 0, 0.05], 0.62, 55, -42),
}
rows = []
for name, (t, d, y, pt) in cams.items():
    img = shot(t, d, pt and pt or 0, pt) if False else shot(t, d, y, pt)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(lab, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
    cv2.putText(lab, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    rows.append(lab)

montage = np.vstack(rows)
cv2.imwrite(OUT + "cam_sheet.png", montage)
print("wrote cam_sheet.png", montage.shape)
env.close()
