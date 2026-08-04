#!/bin/bash
set -e
echo "=== Test 2: cross-protocol FID anchor (IDEAL IQ vs IDEAL-FAM) ==="
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
    print("MISSING after install attempt:", missing); sys.exit(1)
print("all FID dependencies present")
PY

# IQ shard must already be on staging (built off-CHTC from datavault, then uploaded)
IQ_SHARD="/staging/s/sbcook/iq3t_slices.dat"
IQ_META="/staging/s/sbcook/iq3t_slices.json"
FAM_SHARD="/staging/s/sbcook/fam3t_slices.dat"
FAM_META="/staging/s/sbcook/fam3t_slices.json"

IQ_PNG="./iq_png"
FAM_PNG="./fam_png"

echo "--- building 3000 IQ + all FAM magnitude PNGs ---"
python test2_prep.py \
  --iq_shard "$IQ_SHARD"  --iq_meta  "$IQ_META" \
  --fam_shard "$FAM_SHARD" --fam_meta "$FAM_META" \
  --n_iq 3000 --seed 0 --iq_out "$IQ_PNG" --fam_out "$FAM_PNG"

echo "IQ PNGs:  $(ls "$IQ_PNG"  | wc -l)"
echo "FAM PNGs: $(ls "$FAM_PNG" | wc -l)"

echo "--- computing FID (IQ vs FAM) = upper anchor ---"
DEV="cpu"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" && DEV="cuda:0"
python -m pytorch_fid "$IQ_PNG" "$FAM_PNG" --device "$DEV" --batch-size 50

echo "=== Test 2 finished ==="