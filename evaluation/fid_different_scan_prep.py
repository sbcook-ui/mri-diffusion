"""
test2_prep.py  --  cross-protocol FID anchor (Test 2)

Renders:
    A = 3000 random IQ images   (from iq3t_slices.dat)
    B = all FAM images          (from fam3t_slices.dat, ~22164)
both with the EXACT preprocessing in fid_prep.py, so FID(A,B) mirrors your
original gen(3000)-vs-FAM(22k) run -- but with real IDEAL IQ images in place of
generated ones. That number is your "different but plausible distribution"
upper anchor: how far a genuine protocol shift (IDEAL IQ 3D BH vs IDEAL-FAM 2D
FB, same patients/scanner/anatomy) lands in FID units.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image


def mag_uint8(arr2, eps=1e-8):
    """Identical to fid_prep.py: per-image min-max normalized uint8 magnitude."""
    m = np.sqrt(arr2[0] ** 2 + arr2[1] ** 2)
    mn, mx = float(m.min()), float(m.max())
    m = (m - mn) / (mx - mn + eps)
    return (m * 255.0 + 0.5).astype(np.uint8)


def open_shard(shard, meta_path):
    shape = tuple(json.load(open(meta_path))["shape"])
    mm = np.memmap(shard, dtype=np.float32, mode="r", shape=shape)
    return mm, shape[0]


def render(mm, idxs, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    for k, i in enumerate(idxs):
        img = mag_uint8(np.asarray(mm[int(i)]))
        Image.fromarray(img, mode="L").save(
            os.path.join(out_dir, f"{tag}_{int(i):06d}.png"))
        if k % 2000 == 0:
            print(f"  {tag}: wrote {k}/{len(idxs)}", flush=True)
    print(f"wrote {len(idxs)} {tag} PNGs to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq_shard", required=True)
    ap.add_argument("--iq_meta", required=True)
    ap.add_argument("--fam_shard", required=True)
    ap.add_argument("--fam_meta", required=True)
    ap.add_argument("--n_iq", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iq_out", default="./iq_png")
    ap.add_argument("--fam_out", default="./fam_png")
    args = ap.parse_args()

    iq_mm, iq_N = open_shard(args.iq_shard, args.iq_meta)
    fam_mm, fam_N = open_shard(args.fam_shard, args.fam_meta)
    print(f"IQ shard N={iq_N}   FAM shard N={fam_N}", flush=True)

    n_iq = min(args.n_iq, iq_N)
    iq_idx = np.random.default_rng(args.seed).permutation(iq_N)[:n_iq]
    print(f"sampling {n_iq} random IQ images (seed={args.seed})", flush=True)

    render(iq_mm, iq_idx, args.iq_out, "iq")
    render(fam_mm, np.arange(fam_N), args.fam_out, "fam")
    print("prep done.", flush=True)


if __name__ == "__main__":
    main()