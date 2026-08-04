#!/bin/bash
set -e
echo "=== memorization check starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "no gpu (ok, will use cpu)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

GEN="/staging/s/sbcook/generated_100/generated_raw.npy"
SHARD="/staging/s/sbcook/fam3t_slices.dat"
META="/staging/s/sbcook/fam3t_slices.json"
OUT_DIR="/staging/s/sbcook/memorization_out"
mkdir -p "$OUT_DIR"

python memorization_check.py --generated "$GEN" --shard "$SHARD" \
  --shard_meta "$META" --out_dir "$OUT_DIR" --topk 3 --real_batch 512

echo "=== memorization check finished ==="
ls -lh "$OUT_DIR" | head