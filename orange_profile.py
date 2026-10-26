import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/q_yaw_0.png").astype(int)
H, W, _ = img.shape
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
orange = (R > 150) & (G > 70) & (G < 190) & (B < 110)
boxred = (R > 150) & (G < 60) & (B < 60)

# cup column: x with most orange
xs = np.where(orange.any(axis=0))[0]
cx = int(np.median(xs)) if len(xs) else W // 2
print("MARK cup x-range:", (int(xs.min()), int(xs.max())) if len(xs) else None, "cx:", cx)
col_o = np.where(orange[:, cx])[0]
print("MARK orange runs on cx:", end=" ")
runs, s, p = [], None, None
for y in col_o:
    if s is None:
        s = p = y
    elif y > p + 3:
        runs.append((s, p)); s = p = y
    else:
        p = y
if s is not None:
    runs.append((s, p))
print(runs)
col_r = np.where(boxred[:, cx])[0]
print("MARK boxred runs on cx:",
      [(col_r[i], col_r[i]) for i in range(0, len(col_r), max(1, len(col_r) // 6))][:8],
      "total:", len(col_r))
