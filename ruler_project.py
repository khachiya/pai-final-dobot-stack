import numpy as np, pybullet as p, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_90.png")
tip = np.array([0.1505, -0.0001, 0.10])
tgt = [float(tip[0] + 0.02), float(tip[1]), float(tip[2] + 0.03)]
view = np.array(p.computeViewMatrixFromYawPitchRoll(tgt, 0.25, 90, -2, 0, 2)).reshape(4, 4, order="F")
proj = np.array(p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)).reshape(4, 4, order="F")
M = proj @ view

def project(x, y, z):
    v = M @ np.array([x, y, z, 1.0])
    ndc = v[:3] / v[3]
    return int((ndc[0] + 1) * 256), int((1 - ndc[1]) * 256)

print("MARK projection check (z_mm_above_TCP -> px_x,px_y,BGR):")
for dz in [-10, -5, 0, 3, 5, 10, 15, 20, 25, 30, 40]:
    px, py = project(tip[0], tip[1], tip[2] + dz / 1000.0)
    bgr = img[py, px].tolist() if 0 <= px < 512 and 0 <= py < 512 else None
    print(f"MARK z={dz:+d}mm px=({px},{py}) BGR={bgr}")
