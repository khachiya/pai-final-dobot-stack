import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/sv6_b.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
dark = (R + G + B) < 200
ys_r, xs_r = np.where(red)
cx = (int(xs_r.min()) + int(xs_r.max())) // 2
col = dark[:, cx - 15:cx + 16].any(axis=1)
ys_d = np.where(col)[0]
ys_d = ys_d[ys_d < int(ys_r.min())]
print(f"MARK box_cx={cx} box_top={int(ys_r.min())} dark_bottom={int(ys_d.max()) if len(ys_d) else None}", flush=True)
if len(ys_d):
    print(f"MARK vgap_px={int(ys_r.min()) - int(ys_d.max())}", flush=True)
above = dark & (np.arange(img.shape[0])[:, None] < int(ys_r.min()))
ys_a, xs_a = np.where(above)
yb = int(ys_a.max())
blob = above[max(0, yb - 30):yb + 1, :]
_, xb = np.where(blob)
print(f"MARK cup_bottom_row={yb} cup_cx={int(np.median(xb))} dx_px={int(np.median(xb)) - cx}", flush=True)
