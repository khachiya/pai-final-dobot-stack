import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/verify_tcp28.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
H, W, _ = img.shape
# find box x-range via red
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
xs = np.where(red.any(axis=0))[0]
print("MARK red xrange:", int(xs.min()), int(xs.max()), flush=True)
cx = (int(xs.min()) + int(xs.max())) // 2
# detailed column profile every 4px, rows 100-300
out = []
for y in range(100, 300, 4):
    r, g, b = img[y, cx, :]
    s = int(r) + int(g) + int(b)
    if r > 100 and r > 2 * g and r > 2 * b:
        out.append("R")
    elif s < 150:
        out.append("K")
    elif s > 700:
        out.append("W")
    else:
        out.append(".")
print("MARK col profile rows100-296:", "".join(out), flush=True)
print("MARK box width px:", int(xs.max()) - int(xs.min()), flush=True)
