import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_abs.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
dark = (R + G + B) < 200
# dark pixel count per column (which x has the cup?)
per_col = dark.sum(axis=0)
top = np.argsort(per_col)[-6:]
print("MARK top dark columns:", sorted([(int(x), int(per_col[x])) for x in top]), flush=True)
# dark runs on the single darkest column
cx = int(top[np.argmax(per_col[top])])
col = dark[:, cx]
runs, s = [], None
for y in range(512):
    if col[y] and s is None:
        s = y
    if not col[y] and s is not None:
        runs.append((s, y))
        s = None
if s is not None:
    runs.append((s, 512))
print("MARK dark runs on col", cx, ":", runs, flush=True)
