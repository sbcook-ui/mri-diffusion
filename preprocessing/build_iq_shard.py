"""
build_iq_shard.py

Builds an IDEAL IQ 3T training-style shard using the SAME preprocessing as
build_shard.py, so the resulting images are directly comparable to fam3t for FID.

Differences from build_shard.py (only these three):
  1. Restricts to an explicit allowlist of filenames (--list iq_files.txt),
     because the source folder also contains 1.5T IQ files we must exclude.
  2. Writes iq3t_slices.dat / iq3t_slices.json (not fam3t_*).
  3. Records whatever echo count IQ has instead of asserting == 6.
     The 256x256 assertion is kept -- if IQ isn't 256x256, STOP and decide on
     a crop/pad, don't silently change the intensity statistics.

Everything else -- reader, per-image divide-by-max-magnitude normalization,
real/imag channel layout, (slice, echo) pooling -- is identical to build_shard.py.

Usage:
    python build_iq_shard.py /path/to/mat_dir /path/to/out_dir --list iq_files.txt
"""

import sys
import os
import glob
import json
import argparse
import numpy as np

IMG_H = 256
IMG_W = 256


# ---------- readers (identical to build_shard.py) ----------

def read_images_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        grp = None
        for k in f.keys():
            if k.startswith("#"):
                continue
            obj = f[k]
            if isinstance(obj, h5py.Group) and "images" in obj:
                grp = obj
                break
        if grp is None and "images" in f:
            dset = f["images"]
        elif grp is not None:
            dset = grp["images"]
        else:
            raise KeyError(f"no 'images' field found in {path}")

        raw = dset[()]
        if raw.dtype.names and "real" in raw.dtype.names:
            arr = raw["real"] + 1j * raw["imag"]
        else:
            arr = np.asarray(raw)
        arr = np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))
    return arr


def read_images_scipy(path):
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    for k in d.keys():
        if k.startswith("__"):
            continue
        obj = d[k]
        if hasattr(obj, "_fieldnames") and "images" in obj._fieldnames:
            return np.asarray(obj.images)
    raise KeyError(f"no 'images' field found in {path}")


def read_images(path):
    try:
        return read_images_h5(path)
    except Exception:
        return read_images_scipy(path)


# ---------- conversion ----------

def count_samples(files):
    total = 0
    per_file = []
    for p in files:
        arr = read_images(p)
        assert arr.ndim == 4, f"{p}: expected 4D images, got {arr.shape}"
        h, w, n_sl, n_ec = arr.shape
        assert (h, w) == (IMG_H, IMG_W), \
            f"{p}: matrix {h}x{w} != {IMG_H}x{IMG_W} (crop/pad needed -- STOP)"
        # NOTE: echo count recorded, not asserted (IQ may differ from FAM's 6)
        n = n_sl * n_ec
        per_file.append({"file": os.path.basename(p),
                         "slices": int(n_sl), "echoes": int(n_ec), "n": int(n)})
        total += n
        print(f"  {os.path.basename(p)}: {n_sl} slices x {n_ec} echoes = {n}")
    return total, per_file


def build(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dat_path = os.path.join(out_dir, "iq3t_slices.dat")
    json_path = os.path.join(out_dir, "iq3t_slices.json")

    print("Pass 1: counting samples...")
    total_n, per_file = count_samples(files)
    print(f"Total samples N = {total_n}")

    print(f"Pass 2: writing memmap {dat_path}  shape ({total_n}, 2, {IMG_H}, {IMG_W})")
    mm = np.memmap(dat_path, dtype=np.float32, mode="w+",
                   shape=(total_n, 2, IMG_H, IMG_W))

    idx = 0
    for p in files:
        arr = read_images(p)
        h, w, n_sl, n_ec = arr.shape
        for s in range(n_sl):
            for e in range(n_ec):
                img = arr[:, :, s, e]
                mag = np.abs(img).max()
                if mag <= 0 or not np.isfinite(mag):
                    mag = 1.0
                img = img / mag
                mm[idx, 0] = np.real(img).astype(np.float32)
                mm[idx, 1] = np.imag(img).astype(np.float32)
                idx += 1
        print(f"  wrote {os.path.basename(p)} -> running total {idx}/{total_n}")

    mm.flush()
    assert idx == total_n, f"count mismatch: wrote {idx}, expected {total_n}"

    meta = {
        "n": int(total_n),
        "shape": [int(total_n), 2, IMG_H, IMG_W],
        "dtype": "float32",
        "channels": "0=real, 1=imag",
        "normalization": "per-image divide by max magnitude",
        "n_files": len(files),
        "per_file": per_file,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Shard: {dat_path}  Metadata: {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--list", required=True,
                    help="text file of basenames to include (one per line)")
    args = ap.parse_args()

    allow = {ln.strip() for ln in open(args.list) if ln.strip()}
    all_mats = glob.glob(os.path.join(args.mat_dir, "*.mat"))
    files = sorted(p for p in all_mats if os.path.basename(p) in allow)

    missing = allow - {os.path.basename(p) for p in files}
    if missing:
        print(f"WARNING: {len(missing)} allowlisted files not found in {args.mat_dir}:")
        for m in sorted(missing):
            print(f"  missing: {m}")
    print(f"selected {len(files)}/{len(allow)} allowlisted .mat files")
    if not files:
        sys.exit("no matching .mat files -- check mat_dir and --list")
    build(files, args.out_dir)


if __name__ == "__main__":
    main()