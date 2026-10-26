import numpy as np, cv2

for tag in ["high", "low"]:
    img = cv2.imread(f"/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/orx_{tag}.png").astype(int)
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    orange = (R > 150) & (G > 70) & (G < 200) & (B < 120)
    green = (G > 120) & (R < 120) & (B < 120)
    ys_o, xs_o = np.where(orange)
    ys_g = np.where(green.any(axis=1))[0]
    print(f"MARK {tag}: norange={len(ys_o)} orange_y {int(ys_o.min())}-{int(ys_o.max())} "
          f"orange_x {int(xs_o.min())}-{int(xs_o.max())}", flush=True)
    print(f"MARK {tag}: ngreen_markers rows: {sorted([int(y) for y in ys_g[::max(1, len(ys_g)//10)]])}", flush=True)
    # what is directly below orange bottom (rows bottom..bottom+30, cup x-range)?
    yb = int(ys_o.max())
    below = img[yb:yb + 30, int(xs_o.min()):int(xs_o.max()) + 1, :].reshape(-1, 3).mean(axis=0)
    print(f"MARK {tag}: below_orange BGR=({below[0]:.0f},{below[1]:.0f},{below[2]:.0f})", flush=True)
