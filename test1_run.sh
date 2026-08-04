#!/bin/bash
set -e
echo "=== Test 1: patient-hold-out FID floor ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "no gpu (ok, will use cpu -- slower)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# ---- FID dependencies (same as your fid_run.sh) ----
echo "--- checking FID dependencies ---"
python -c "import PIL"          2>/dev/null || pip install --no-cache-dir pillow
python -c "import scipy"        2>/dev/null || pip install --no-cache-dir scipy
python -c "import torchvision"  2>/dev/null || pip install --no-cache-dir --no-deps torchvision==0.19.0
python -c "import pytorch_fid"  2>/dev/null || pip install --no-cache-dir --no-deps pytorch-fid

python - <<'PY'
import importlib, sys
missing = [m for m in ("PIL", "scipy", "torchvision", "pytorch_fid")
           if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING after install attempt:", missing)
    sys.exit(1)
print("all FID dependencies present")
PY

SHARD="/staging/s/sbcook/fam3t_slices.dat"
META="/staging/s/sbcook/fam3t_slices.json"

# PNGs go to LOCAL job scratch -- we only keep the FID number.
A_PNG="./real_holdout"
B_PNG="./real_rest"

echo "--- building A (held-out) and B (rest) magnitude PNGs ---"
python test1_prep.py --shard "$SHARD" --shard_meta "$META" \
  --a_out "$A_PNG" --b_out "$B_PNG"

echo "A (held-out) PNGs: $(ls "$A_PNG" | wc -l)"
echo "B (rest)     PNGs: $(ls "$B_PNG" | wc -l)"

echo "--- computing FID (held-out vs rest) = the floor at ~3k ---"
DEV="cpu"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" && DEV="cuda:0"
python -m pytorch_fid "$A_PNG" "$B_PNG" --device "$DEV" --batch-size 50

echo "=== Test 1 finished ==="