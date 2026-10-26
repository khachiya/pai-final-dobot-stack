import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_fixed.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
for name, m in [("orange", orange), ("red", red)]:
    ys, xs = np.where(m)
    if len(ys):
        print(f"MARK {name}: rows {int(ys.min())}-{int(ys.max())} cols {int(xs.min())}-{int(xs.max())} n={len(ys)}", flush=True)
    else:
        print(f"MARK {name}: none", flush=True)
# red pixels below orange bottom?
ys_o = np.where(orange.any(axis=1))[0]
ys_r = np.where(red.any(axis=1))[0]
below = ys_r[ys_r > ys_o.max()] if len(ys_o) and len(ys_r) else []
print("MARK red rows below orange_bottom:", int(below.min()) if len(below) else None, flush=True)
