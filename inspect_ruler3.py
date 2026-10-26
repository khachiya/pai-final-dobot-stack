import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_abs.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
# green markers x positions
green = (G > 120) & (R < 120) & (B < 120)
xs = np.where(green.any(axis=0))[0]
print("MARK green xrange:", int(xs.min()), int(xs.max()), flush=True)
# exact brightness around cup bottom (x=276..282), rows 180-210
col = img[180:211, 276:283, :].mean(axis=1)
for i, y in enumerate(range(180, 211, 5)):
    print(f"MARK row{y}:", [int(v) for v in img[y, 276:283, :].mean(axis=0)], flush=True)
