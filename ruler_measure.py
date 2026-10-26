import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_90.png")
H, W, _ = img.shape
R = img[:, :, 2].astype(int)
G = img[:, :, 1].astype(int)
B = img[:, :, 0].astype(int)
red = (R > 120) & (R > G + 40) & (R > B + 40)
dark = (R + G + B) < 150

# red cube clusters (columns around x=243..269 where cubes are)
sub = red[:, 230:285]
ys = np.where(sub.any(axis=1))[0]
groups, cur = [], [ys[0]]
for y in ys[1:]:
    if y - cur[-1] <= 4:
        cur.append(y)
    else:
        groups.append(cur); cur = [y]
groups.append(cur)
print("MARK red groups (y0-y1):", [(g[0], g[-1]) for g in groups])

# cup axis: darkest long vertical structure left of cubes (x in 100..230)
best, bestcol = 0, 0
for x in range(100, 230):
    col = dark[:, x]
    # longest dark run
    run, mx = 0, 0
    for v in col:
        run = run + 1 if v else 0
        mx = max(mx, run)
    if mx > best:
        best, bestcol = mx, x
col = dark[:, bestcol]
ys2 = np.where(col)[0]
print(f"MARK cup col x={bestcol} dark from {ys2.min()} to {ys2.max()}")
# cup bottom: last dark row before a bright gap (>8 white rows)
bottom = ys2.max()
for i in range(len(ys2) - 1, 0, -1):
    if ys2[i - 1] < ys2[i] - 8:
        bottom = ys2[i]
        break
print("MARK cup_bottom_row:", bottom)
