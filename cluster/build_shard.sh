#!/bin/bash
# build_shard.sh - runs inside the container on CHTC.
# Reads the 115 .mat files from staging, writes the shard back to staging.

set -e  # stop on any error

echo "=== shard build job starting ==="
echo "hostname: $(hostname)"
echo "python: $(python --version)"

# sanity: confirm the packages we added are present
python -c "import numpy, scipy, h5py; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| h5py ok')"

MAT_DIR="/staging/s/sbcook/fam3t_local"
OUT_DIR="/staging/s/sbcook"

echo "input dir:  $MAT_DIR"
echo "output dir: $OUT_DIR"
echo "num .mat files: $(ls "$MAT_DIR"/*.mat | wc -l)"

# run the converter (build_shard.py is transferred with the job)
python build_shard.py "$MAT_DIR" "$OUT_DIR"

echo "=== shard build finished ==="
ls -lh "$OUT_DIR"/fam3t_slices.dat "$OUT_DIR"/fam3t_slices.json
