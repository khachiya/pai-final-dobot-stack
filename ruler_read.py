import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_side.png").astype(int)
H, W, _ = img.shape
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
green = (G > 120) & (R < 150) & (B < 150)
dark = (R + G + B) < 180

# ruler dashes: green clusters (whole image), group by row
ys = np.where(green.any(axis=1))[0]
groups, cur = [], [ys[0]]
for y in ys[1:]:
    if y - cur[-1] <= 6:
        cur.append(y)
    else:
        groups.append(cur); cur = [y]
groups.append(cur)
print("MARK green dash rows:", [(g[0] + g[-1]) // 2 for g in groups])

# cup axis: use image center x (target was tip) -> dark runs
cx = W // 2
col = dark[:, cx]
ys2 = np.where(col)[0]
runs, s, prev = [], ys2[0], ys2[0]
for y in ys2[1:]:
    if y > prev + 4:
        runs.append((s, prev)); s = y
    prev = y
runs.append((s, prev))
print("MARK dark runs on center col:", runs)
