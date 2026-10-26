"""Inject per-visual materials into the Dobot URDF (meshes ship without colors)."""
import re

PATH = "/home/khach/pai-final/models/magician/dobot_magician.urdf"
src = open(PATH).read()

colors = {
    "magician_link_base": (0.85, 0.85, 0.83, 1),
    "magician_link_1":   (0.92, 0.92, 0.90, 1),
    "magician_link_2":   (0.94, 0.94, 0.93, 1),
    "magician_link_3":   (0.94, 0.94, 0.93, 1),
    "magician_link_4":   (0.25, 0.25, 0.26, 1),
    "magician_link_5":   (0.20, 0.20, 0.21, 1),
}

out = src
for link, c in colors.items():
    m = re.search(r'<link name="' + link + r'">.*?</link>', out, re.S)
    if not m:
        print(link, "NOT FOUND")
        continue
    block = m.group(0)
    mat = ('<material name="m_' + link + '"><color rgba="'
           + f"{c[0]} {c[1]} {c[2]} {c[3]}" + '"/></material>')
    if "<visual>" in block:
        newblock = block.replace("<visual>", "<visual>\n      " + mat, 1)
        out = out.replace(block, newblock)
        print(link, "ok")
    else:
        print(link, "no <visual>")

open(PATH, "w").write(out)
print("materials now:", out.count("<material"))
