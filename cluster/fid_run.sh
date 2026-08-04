#!/bin/bash
set -e
echo "=== FID computation starting ==="
echo "hostname: $(hostname)"
nvidia-smi || echo "no gpu (ok, will use cpu -- slower)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# ---- make sure the FID dependencies are available ----
# The image has torch + numpy. FID also needs torchvision (matched to torch 2.4.0),
# scipy, pillow, and the pytorch-fid package itself. Install only what's missing.
# --no-deps on torchvision/pytorch-fid avoids pulling a conflicting torch build.
echo "--- checking FID dependencies ---"
python -c "import PIL"          2>/dev/null || pip install --no-cache-dir pillow
python -c "import scipy"        2>/dev/null || pip install --no-cache-dir scipy
python -c "import torchvision"  2>/dev/null || pip install --no-cache-dir --no-deps torchvision==0.19.0
python -c "import pytorch_fid"  2>/dev/null || pip install --no-cache-dir --no-deps pytorch-fid

# hard-fail early with a clear message if anything still isn't importable
python - <<'PY'
import importlib, sys
missing = [m for m in ("PIL", "scipy", "torchvision", "pytorch_fid")
           if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING after install attempt:", missing)
    print("The execute node likely has no internet for pip. "
          "Add these to the container image and rebuild, then resubmit.")
    sys.exit(1)
print("all FID dependencies present")
PY

GEN="/staging/s/sbcook/generated_100/generated_raw.npy"
SHARD="/staging/s/sbcook/fam3t_slices.dat"
META="/staging/s/sbcook/fam3t_slices.json"

# PNGs go to LOCAL job scratch (fast, auto-cleaned) -- we only keep the FID number.
GEN_PNG="./gen_png"
REAL_PNG="./real_png"

echo "--- building magnitude PNGs ---"
python fid_prep.py --generated "$GEN" --shard "$SHARD" --shard_meta "$META" \
  --gen_out "$GEN_PNG" --real_out "$REAL_PNG"

echo "real PNGs:      $(ls "$REAL_PNG" | wc -l)"
echo "generated PNGs: $(ls "$GEN_PNG" | wc -l)"

echo "--- computing FID (real vs generated) ---"
DEV="cpu"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" && DEV="cuda:0"
python -m pytorch_fid "$REAL_PNG" "$GEN_PNG" --device "$DEV" --batch-size 50

echo "=== FID computation finished ==="