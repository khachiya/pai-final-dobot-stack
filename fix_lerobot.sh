#!/usr/bin/env bash
set -uo pipefail
cd /home/khach/pai-final
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

echo "== stub evdev (needs kernel headers; not present in this WSL) =="
rm -rf /home/khach/stubs && mkdir -p /home/khach/stubs/src/evdev_stub/evdev /home/khach/stubs/wheels
cat > /home/khach/stubs/src/evdev_stub/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "evdev"
version = "9.9.9"
description = "stub: keyboard teleop not used on this headless server"
requires-python = ">=3.8"
EOF
cat > /home/khach/stubs/src/evdev_stub/evdev/__init__.py <<'EOF'
raise ImportError("evdev stub installed; input-device teleoperation is unavailable")
EOF
cd /home/khach/stubs/src/evdev_stub && uv build --wheel -q -o /home/khach/stubs/wheels && cd /home/khach/pai-final

echo "== pre-install torch cu128 within lerobot constraint (<2.11) =="
uv pip uninstall torch torchvision || true
uv pip install "torch<2.11" "torchvision<0.26" --index-url https://download.pytorch.org/whl/cu128

echo "== lerobot 0.4.4 (py3.11-compatible; latest needs py>=3.12) =="
uv pip install "lerobot[smolvla]==0.4.4" --find-links /home/khach/stubs/wheels || { echo LR_FAIL; exit 1; }

echo "== ensure torch still cu128 build (lerobot must not have replaced it) =="
python -c "import torch,sys; print('torch', torch.__version__); sys.exit(0 if '+cu12' in torch.__version__ else 1)" || uv pip install --reinstall "torch<2.11" "torchvision<0.26" --index-url https://download.pytorch.org/whl/cu128

echo "== pybullet (cp311 wheel available) =="
uv pip install pybullet

echo "== verify =="
python - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("matmul ok", float(x.sum()))
import pybullet as p
print("pybullet", p.getVersion())
import lerobot; print("lerobot", getattr(lerobot, "__version__", "?"))
for path in ("lerobot.common.policies.act.modeling_act", "lerobot.policies.act.modeling_act"):
    try:
        importlib.import_module(path); print("ACT module OK:", path)
    except Exception as e:
        print("no:", path, type(e).__name__)
EOF
echo FIX2_DONE
