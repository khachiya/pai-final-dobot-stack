import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orange_ruler.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
# shadowed orange: R dominant but dimmer
for y in [185, 189, 193, 197, 201, 205, 210]:
    patch = img[y, 240:295, :]
    print(f"MARK row{y}: B={patch[:, 0].mean():.0f} G={patch[:, 1].mean():.0f} R={patch[:, 2].mean():.0f}",
          flush=True)
