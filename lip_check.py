import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_fixed.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
for y in [234, 238, 242, 246, 250, 254]:
    patch = img[y, 240:275, :]
    print(f"MARK row{y}: B={patch[:, 0].mean():.0f} G={patch[:, 1].mean():.0f} R={patch[:, 2].mean():.0f}",
          flush=True)
