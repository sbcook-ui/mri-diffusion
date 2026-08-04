#!/bin/bash
# gputest.sh - runs inside the container on CHTC.
# Full-size ADM model, only ~300 steps, to validate the GPU training path.

set -e

echo "=== GPU test job starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

SHARD_DIR="/staging/s/sbcook"          # where fam3t_slices.dat/.json live
CKPT_DIR="/staging/s/sbcook/checkpoints_gputest"
mkdir -p "$CKPT_DIR"

echo "shard dir: $SHARD_DIR"
echo "ckpt dir:  $CKPT_DIR"

# run the short GPU test
python train.py --gputest --shard_dir "$SHARD_DIR" --ckpt_dir "$CKPT_DIR"

echo "=== GPU test finished ==="
ls -lh "$CKPT_DIR"