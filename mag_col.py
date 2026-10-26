import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/mag2.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
# cup bottom region: x 228-298, look downward for first red-ish row
for y0 in [150, 160, 180, 200, 220, 240, 260]:
    patch = img[y0:y0 + 8, 228:298, :]
    r, g, b = patch[:, :, 2].mean(), patch[:, :, 1].mean(), patch[:, :, 0].mean()
    print(f"MARK row{y0}: R={r:.0f} G={g:.0f} B={b:.0f}", flush=True)
