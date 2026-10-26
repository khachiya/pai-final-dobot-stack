import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/sv5_a.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
H, W, _ = img.shape
print("MARK shape:", H, W, flush=True)
# box: red. cup: black/dark. overlay text at top (~100px) - exclude top region
red = (R > 100) & (R > 2 * G) & (R > 2 * B)
dark = (R + G + B) < 200
ys_r, xs_r = np.where(red)
ys_d, xs_d = np.where(dark)
print(f"MARK red rows {int(ys_r.min())}-{int(ys_r.max())} cols {int(xs_r.min())}-{int(xs_r.max())} n={len(ys_r)}", flush=True)
print(f"MARK dark rows {int(ys_d.min())}-{int(ys_d.max())} cols {int(xs_d.min())}-{int(xs_d.max())} n={len(ys_d)}", flush=True)
# cup bottom: dark pixels above box top, near box x-center
bxc = (int(xs_r.min()) + int(xs_r.max())) // 2
col = dark[:, bxc - 15:bxc + 16].any(axis=1)
ys_c = np.where(col)[0]
ys_c = ys_c[ys_c < int(ys_r.min())]
print(f"MARK box_cx={bxc} dark_bottom_above_box={int(ys_c.max()) if len(ys_c) else None} "
      f"box_top={int(ys_r.min())}", flush=True)
if len(ys_c):
    print(f"MARK vgap_px={int(ys_r.min()) - int(ys_c.max())}", flush=True)
# cup x-center (dark mass above box) vs box x-center
above = dark & (np.arange(H)[:, None] < int(ys_r.min()))
ys_a, xs_a = np.where(above)
if len(ys_a):
    # lowest dark blob = cup: take bottom 30 rows of dark-above-box
    yb = int(ys_a.max())
    blob = above[max(0, yb - 30):yb + 1, :]
    _, xs_b = np.where(blob)
    print(f"MARK cup_bottom_row={yb} cup_cx={int(np.median(xs_b))} box_cx={bxc} "
          f"dx_px={int(np.median(xs_b)) - bxc}", flush=True)
