#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
export PATH="$HOME/.local/bin:$PATH"

echo "== rebuild venv with python 3.11 (pybullet has cp311 wheels; no compiler needed) =="
uv python install 3.11
uv venv --clear --python 3.11 .venv
source .venv/bin/activate
python --version

echo "== lerobot + pybullet + av =="
uv pip install av "lerobot[smolvla]" pybullet

echo "== force torch for CUDA 12.x driver (after lerobot) =="
uv pip uninstall torch torchvision || true
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "== verify =="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("matmul ok", float(x.sum()))
import pybullet as p
print("pybullet", p.getVersion())
import lerobot
print("lerobot OK")
EOF
echo FIX_DONE
