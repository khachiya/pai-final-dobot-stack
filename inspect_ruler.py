import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_abs.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
dark = (R + G + B) < 200
ys, xs = np.where(dark)
print("MARK ndark:", len(ys), flush=True)
if len(ys):
    print("MARK yrange:", int(ys.min()), int(ys.max()), "xrange:", int(xs.min()), int(xs.max()), flush=True)
col = img[:, 256, :].mean(axis=1)
print("MARK center col rows100-300:", [int(v) for v in col[100:300:20]], flush=True)
print("MARK img shape:", img.shape, "corner px:", img[0, 0, :].tolist(), flush=True)
