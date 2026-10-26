import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/dark_bg.png").astype(int)
H, W, _ = img.shape
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
dark = (R + G + B) < 150
red = (R > 150) & (G < 60) & (B < 60)

n, lab, stats, cent = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
print("MARK dark components:", n - 1)
best = None
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area > 200 and y < 200 and 100 < x < 400:
        if best is None or (y + h) > best[0]:
            best = (y + h, x, x + w, area)
print("MARK nozzle candidate (bottom,x0,x1,n):", best)
if best:
    cx = (best[1] + best[2]) // 2
    col = red[:, cx]
    ys2 = np.where(col)[0]
    ys2 = ys2[ys2 > best[0]]
    top = int(ys2.min()) if len(ys2) else None
    print(f"MARK nozzle_bottom={best[0]} redtop_below={top} gap_px={top-best[0] if top else None} cx={cx}")
    # scale: red box height at a column away from nozzle
    col2 = red[:, 100]
    ys3 = np.where(col2)[0]
    print("MARK boxcol_red_span:", (int(ys3.min()), int(ys3.max())) if len(ys3) else None)
