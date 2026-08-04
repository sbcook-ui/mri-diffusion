#!/bin/bash
# run100k.sh - runs inside the container on CHTC.
# Full-size ADM model, 100k steps, checkpoints every 10k to staging.

set -e

echo "=== 100k training run starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

SHARD_DIR="/staging/s/sbcook"
CKPT_DIR="/staging/s/sbcook/checkpoints_100k"
mkdir -p "$CKPT_DIR"

echo "shard dir: $SHARD_DIR"
echo "ckpt dir:  $CKPT_DIR"

# If a checkpoint already exists (from a previous interrupted run), resume from the latest.
LATEST=$(ls -1 "$CKPT_DIR"/model_*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "found existing checkpoint, resuming from: $LATEST"
    python train.py --run100k --shard_dir "$SHARD_DIR" --ckpt_dir "$CKPT_DIR" --resume "$LATEST"
else
    echo "no existing checkpoint, starting fresh"
    python train.py --run100k --shard_dir "$SHARD_DIR" --ckpt_dir "$CKPT_DIR"
fi

echo "=== 100k training run finished ==="
ls -lh "$CKPT_DIR"