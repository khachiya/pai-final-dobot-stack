import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_90.png").astype(int)
H, W, _ = img.shape
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
dark = (R + G + B) < 150

cands = []
for x in range(100, 245):
    col = dark[:, x]
    # find dark runs
    runs, s = [], None
    for y in range(H):
        if col[y] and s is None:
            s = y
        elif not col[y] and s is not None:
            runs.append((s, y - 1)); s = None
    if s is not None:
        runs.append((s, H - 1))
    for a, b in runs:
        ln = b - a + 1
        if 20 <= ln <= 70 and b >= 180:
            # white below?
            below = (~col[b + 1:b + 25]).sum() if b + 25 < H else 0
            cands.append((x, a, b, below))
print("MARK cands(x,a,b,whitebelow):", cands[:40])
