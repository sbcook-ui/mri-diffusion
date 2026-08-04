#!/bin/bash
set -e
echo "=== generate 100 job starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

CKPT="/staging/s/sbcook/checkpoints_100k/model_090000.pt"
OUT_DIR="/staging/s/sbcook/generated_100"
mkdir -p "$OUT_DIR"

echo "checkpoint: $CKPT"
echo "output dir: $OUT_DIR"
sed -i 's/\r$//' gen100.sh memcheck.sh
python sample.py --ckpt "$CKPT" --n 100 --out_dir "$OUT_DIR" --batch 10

echo "=== generation finished ==="
ls -lh "$OUT_DIR" | head