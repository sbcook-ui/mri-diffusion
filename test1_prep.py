"""
test1_prep.py  --  patient-hold-out FID floor (Test 1)

Splits the training shard into two disjoint sets by WHOLE PATIENT:
    A = held-out patients   (~3090 images)
    B = everyone else        (~19074 images)
then renders both to magnitude PNGs using the EXACT preprocessing in
fid_prep.py, so the resulting FID is on the same scale as your 34.

The split is derived purely from fam3t_slices.json's `per_file` list: files
were written contiguously in order, so a cumulative sum of each file's `n`
gives that file's row range in the memmap. Each block is assigned to A or B by
the AnonID parsed from its filename. No .mat files needed.

Run pytorch-fid on the two output folders afterwards (see test1_run.sh).
"""

import os
import re
import json
import argparse
import numpy as np
from PIL import Image

# 15 ordinary axial patients held out (~3090 images). Edit if you reselect.
HOLDOUT = {
    "Anon004", "Anon019", "Anon040", "Anon061", "Anon075", "Anon093",
    "Anon115", "Anon137", "Anon152", "Anon168", "Anon187", "Anon205",
    "Anon220", "Anon234", "Anon244",
}


def mag_uint8(arr2, eps=1e-8):
    """arr2: (2,H,W) real/imag -> per-image min-max normalized uint8 magnitude.
    Identical to fid_prep.py so the FID is comparable to your existing number."""
    m = np.sqrt(arr2[0] ** 2 + arr2[1] ** 2)
    mn, mx = float(m.min()), float(m.max())
    m = (m - mn) / (mx - mn + eps)
    return (m * 255.0 + 0.5).astype(np.uint8)


def build_split(meta_path):
    """Return (idx_A, idx_B) row-index arrays from the per_file manifest."""
    meta = json.load(open(meta_path))
    shape = tuple(meta["shape"])                     # (N,2,H,W)
    idx_A, idx_B = [], []
    off = 0
    for f in meta["per_file"]:
        n = int(f["n"])
        aid = re.search(r"Anon\d+", f["file"]).group(0)
        (idx_A if aid in HOLDOUT else idx_B).extend(range(off, off + n))
        off += n
    assert off == shape[0], f"manifest rows {off} != shape[0] {shape[0]}"
    assert not (set(idx_A) & set(idx_B)), "A/B overlap -- leakage!"
    return shape, np.array(idx_A), np.array(idx_B)


def render(mm, idxs, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    for k, i in enumerate(idxs):
        img = mag_uint8(np.asarray(mm[i]))
        Image.fromarray(img, mode="L").save(
            os.path.join(out_dir, f"{tag}_{int(i):06d}.png"))
        if k % 2000 == 0:
            print(f"  {tag}: wrote {k}/{len(idxs)}", flush=True)
    print(f"wrote {len(idxs)} {tag} PNGs to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--shard_meta", required=True)
    ap.add_argument("--a_out", default="./real_holdout")   # set A
    ap.add_argument("--b_out", default="./real_rest")      # set B
    args = ap.parse_args()

    shape, idx_A, idx_B = build_split(args.shard_meta)
    print(f"held-out patients: {sorted(HOLDOUT)}")
    print(f"A (held-out): {len(idx_A)}   B (rest): {len(idx_B)}   "
          f"total: {len(idx_A) + len(idx_B)} (expect {shape[0]})", flush=True)

    mm = np.memmap(args.shard, dtype=np.float32, mode="r", shape=shape)
    render(mm, idx_A, args.a_out, "hold")
    render(mm, idx_B, args.b_out, "rest")
    print("prep done.", flush=True)


if __name__ == "__main__":
    main()