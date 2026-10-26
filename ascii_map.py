import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/ruler_side.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
green = (G > 120) & (R < 150) & (B < 150)
dark = (R + G + B) < 180
red = (R > G + 40) & (R > B + 40)
h, w, _ = img.shape
for yy in range(0, h, 32):
    row = ""
    for xx in range(0, w, 32):
        blk_g = green[yy:yy + 32, xx:xx + 32].mean()
        blk_d = dark[yy:yy + 32, xx:xx + 32].mean()
        blk_r = red[yy:yy + 32, xx:xx + 32].mean()
        if blk_g > 0.15:
            row += "G"
        elif blk_r > 0.15:
            row += "R"
        elif blk_d > 0.6:
            row += "#"
        elif blk_d > 0.2:
            row += "+"
        else:
            row += "."
    print(f"MARK {yy:3d} {row}")
