#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
source .venv/bin/activate
rm -rf data/smoke
python gen_data.py --episodes 4 --repo-id local/smoke --root /home/khach/pai-final/data/smoke --seed 7
echo ===LS===
find data/smoke -maxdepth 3 | head -25
