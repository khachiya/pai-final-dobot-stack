import numpy as np, cv2

img = cv2.imread("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/mag2.png").astype(int)
R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
mag = (R > 120) & (B > 100) & (G < 170) & ((R + B - G) > 120)
red = (R > 150) & (G < 60) & (B < 60)
redloose = (R > 100) & (R > 2 * G) & (R > 2 * B)
print("MARK nmag:", int(mag.sum()), "nred:", int(red.sum()), "nredloose:", int(redloose.sum()), flush=True)
if int(mag.sum()) and int(red.sum()):
    ym = np.where(mag.any(axis=1))[0]
    yr = np.where(red.any(axis=1))[0]
    print("MARK mag_y:", int(ym.min()), int(ym.max()), "red_y:", int(yr.min()), int(yr.max()),
          "gap:", int(yr.min()) - int(ym.max()), flush=True)
