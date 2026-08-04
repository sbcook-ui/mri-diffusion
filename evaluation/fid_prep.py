"""
fid_prep.py

Prepares images for FID computation, following the recipe in Appendix B.6 of the
Measurement Score-Based Diffusion Model paper (Park et al., arXiv:2505.11853):
complex MRI -> magnitude -> single grayscale channel. (pytorch-fid loads each PNG
with .convert("RGB"), which replicates the single channel to 3 channels, exactly
matching the paper's "replicate magnitude three times" step -- so we save grayscale
and let pytorch-fid do the replication.)

Both generated and real slices are per-image min-max normalized to [0,255] so the
two sets are on an identical intensity scale before Inception features are extracted.

Writes PNGs to local job scratch (not staging) since they are only intermediate
inputs to pytorch-fid; the FID number itself is what you keep (it prints to stdout).

Inputs:
    --generated   path to generated_raw.npy   (n, 2, H, W) float32  (real,imag)
    --shard       path to fam3t_slices.dat     memmap (N, 2, H, W) float32
    --shard_meta  path to fam3t_slices.json
    --gen_out     folder to write generated PNGs   (default: ./gen_png)
    --real_out    folder to write real PNGs        (default: ./real_png)
    --max_real    optionally cap the number of real slices (default: all)
"""

import os
import argparse
import json
import numpy as np
from PIL import Image


def mag_uint8(arr2, eps=1e-8):
    """arr2: (2,H,W) real/imag -> per-image min-max normalized uint8 magnitude (H,W)."""
    m = np.sqrt(arr2[0] ** 2 + arr2[1] ** 2)
    mn, mx = float(m.min()), float(m.max())
    m = (m - mn) / (mx - mn + eps)
    return (m * 255.0 + 0.5).astype(np.uint8)


def save_generated(gen_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = np.load(gen_path)                       # (n,2,H,W)
    n = g.shape[0]
    for i in range(n):
        img = mag_uint8(g[i])
        Image.fromarray(img, mode="L").save(os.path.join(out_dir, f"gen_{i:05d}.png"))
    print(f"wrote {n} generated PNGs to {out_dir}", flush=True)
    return n


def save_reals(shard, meta_path, out_dir, max_real=None):
    os.makedirs(out_dir, exist_ok=True)
    shape = tuple(json.load(open(meta_path))["shape"])   # (N,2,H,W)
    N = shape[0]
    if max_real is not None:
        N = min(N, max_real)
    mm = np.memmap(shard, dtype=np.float32, mode="r", shape=shape)
    for i in range(N):
        img = mag_uint8(np.asarray(mm[i]))
        Image.fromarray(img, mode="L").save(os.path.join(out_dir, f"real_{i:06d}.png"))
        if i % 2000 == 0:
            print(f"  wrote real {i}/{N}", flush=True)
    print(f"wrote {N} real PNGs to {out_dir}", flush=True)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--shard_meta", required=True)
    ap.add_argument("--gen_out", default="./gen_png")
    ap.add_argument("--real_out", default="./real_png")
    ap.add_argument("--max_real", type=int, default=None)
    args = ap.parse_args()

    n_gen = save_generated(args.generated, args.gen_out)
    n_real = save_reals(args.shard, args.shard_meta, args.real_out, args.max_real)
    print(f"\nprep done: {n_gen} generated, {n_real} real", flush=True)


if __name__ == "__main__":
    main()