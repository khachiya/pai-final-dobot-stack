import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/mag2.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
mag = (R > 120) & (B > 100) & ((R - G) > 60) & ((B - G) > 40)
red = (R > 150) & (R > 2 * G + 20) & (R > 2 * B + 20)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mag.astype(np.uint8), 8)
best = None
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area > 300 and 150 < x + w / 2 < 360 and y < 300:
        if best is None or area > best[0]:
            best = (area, y, y + h, x, x + w)
print("MARK cup blob (area,top,bottom,x0,x1):", best, flush=True)
ys = np.where(red.any(axis=1))[0]
runs, s, p = [], ys[0], ys[0]
for y in ys[1:]:
    if y > p + 4:
        runs.append((s, p)); s = y
    p = y
runs.append((s, p))
runs = [r for r in runs if r[1] - r[0] > 5]
print("MARK red runs:", runs[:6], flush=True)
