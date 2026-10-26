import numpy as np, cv2

for tag in ["sv6_b"]:
    img = cv2.imread(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/{tag}.png").astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red = (R > 100) & (R > 2 * G) & (R > 2 * B)
    dark = (R + G + B) < 200
    ys_r = np.where(red.any(axis=1))[0]
    print(f"MARK {tag}: red_y={(int(ys_r.min()), int(ys_r.max())) if len(ys_r) else None} "
          f"ndark={int(dark.sum())} mean_brightness={img.mean():.0f}", flush=True)
