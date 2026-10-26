import numpy as np, pybullet as p, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/markers.png").astype(int)
H, W, _ = img.shape
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]

tip = np.array([0.1509, -0.0001, 0.10])
tgt = [float(tip[0]), float(tip[1]), float(tip[2] + 0.015)]
view = np.array(p.computeViewMatrixFromYawPitchRoll(tgt, 0.25, 90, -2, 0, 2)).reshape(4, 4, order="F")
proj = np.array(p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)).reshape(4, 4, order="F")
M = proj @ view

def project(x, y, z):
    v = M @ np.array([x, y, z, 1.0])
    ndc = v[:3] / v[3]
    return int((ndc[0] + 1) * W / 2), int((1 - ndc[1]) * H / 2)

# projected rows for candidate heights on cup axis
print("MARK proj rows (h_mm -> px,py):")
rows = {}
for h in [15, 18, 20, 22, 24, 26, 28, 30, 32, 35, 40]:
    px, py = project(tip[0], tip[1], tip[2] + h / 1000.0)
    rows[h] = py
    print(f"MARK h={h}mm -> ({px},{py})")

# dark structure excluding green markers
green = (G > 150) & (R < 150) & (B < 150)
dark = ((R + G + B) < 180) & (~green)
ax = project(tip[0], tip[1], tip[2])[0]
print("MARK axis x:", ax)
best = None
for x in range(ax - 15, ax + 16):
    col = dark[:, x]
    ys = np.where(col)[0]
    if len(ys) == 0:
        continue
    # longest run
    runs, s = [], ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y > prev + 3:
            runs.append((s, prev)); s = y
        prev = y
    runs.append((s, prev))
    for a, b in runs:
        if b - a >= 8 and a > 50:
            if best is None or b > best[1]:
                best = (x, b, a)
print("MARK longest dark run bottom (x,bottom,top):", best)
