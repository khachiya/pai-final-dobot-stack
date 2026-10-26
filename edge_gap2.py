import numpy as np, cv2

for name in ["repro", "yaw_270", "sv3_b", "sv2_a", "press_pose"]:
    try:
        img = cv2.imread(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/{name}.png").astype(int)
    except Exception as e:
        print("MARK", name, "missing"); continue
    if img is None:
        print("MARK", name, "missing"); continue
    H, W, _ = img.shape
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    red = (R > G + 40) & (R > B + 40)
    ys, xs = np.where(red)
    if len(ys) == 0:
        print("MARK", name, "no red"); continue
    cx = (xs.min() + xs.max()) // 2
    colred = red[:, cx]
    rtop = np.where(colred)[0].min()
    nonwhite = (R + G + B)[:, cx] < 700
    nonred = ~colred
    obj = nonwhite & nonred
    # lowest object run >=3px tall, scanning up from rtop
    best = None
    y = rtop - 1
    while y > max(0, rtop - 200):
        if obj[y] and obj[max(0, y - 2)]:
            best = y
            break
        y -= 1
    print(f"MARK {name}: redtop={rtop} objbottom={best} gap_px={rtop - best if best else None}")
