import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/chan.png").astype(int)
c0, c1, c2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
mx = np.maximum(np.maximum(c0, c1), c2)
mn = np.minimum(np.minimum(c0, c1), c2)
sat = (mx - mn) > 100
for i, w in [(0, (c0 == mx) & sat), (1, (c1 == mx) & sat), (2, (c2 == mx) & sat)]:
    ys = np.where(w)[0]
    print(f"MARK ch{i} wins: n={int(w.sum())}",
          f"yrange={(int(ys.min()), int(ys.max())) if len(ys) else None}")
