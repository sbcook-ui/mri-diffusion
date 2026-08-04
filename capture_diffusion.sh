#!/bin/bash
set -e
echo "=== diffusion capture starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "no gpu (ok, will use cpu -- slower)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

CKPT="/staging/s/sbcook/checkpoints/model_000200.pt"   # <-- set to your checkpoint on staging
OUT_DIR="/staging/s/sbcook/diffusion_capture"
mkdir -p "$OUT_DIR"

python capture_diffusion.py --ckpt "$CKPT" --out_dir "$OUT_DIR" --seed 0

echo "=== diffusion capture finished ==="
ls -lh "$OUT_DIR"