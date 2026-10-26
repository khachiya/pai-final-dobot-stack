import numpy as np, cv2

for name in ["repro", "yaw_270"]:
    img = cv2.imread(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/{name}.png").astype(int)
    H, W, _ = img.shape
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red = (R > G + 40) & (R > B + 40)
    ys, xs = np.where(red)
    print(f"MARK {name}: red yrange {ys.min()}-{ys.max()} xrange {xs.min()}-{xs.max()}")
    cx = (xs.min() + xs.max()) // 2
    colsum = img[:, cx, :].sum(axis=1)
    # box top = first red row scanning down... use red mask top in this column
    colred = red[:, cx]
    rtop = np.where(colred)[0].min()
    # scan upward from rtop-1: first non-white (sum<700) sustained 2+ px
    obj = None
    for y in range(rtop - 1, max(0, rtop - 150), -1):
        if colsum[y] < 700 and colsum[y - 1] < 700:
            obj = y
            break
    print(f"MARK {name}: col x={cx} redtop={rtop} lowestobj={obj} gap_px={rtop - obj if obj else None}")
