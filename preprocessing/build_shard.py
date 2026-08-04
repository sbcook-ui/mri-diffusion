"""
build_shard.py

Converts all IDEAL-FAM 3T .mat files into ONE memory-mapped training shard
for unconditional diffusion training.

Output:
    fam3t_slices.dat   -> float32 memmap, shape (N, 2, 256, 256)
                          channel 0 = real, channel 1 = imag
    fam3t_slices.json  -> metadata (N, shape, per-file counts, normalization)

Each training sample = one (slice, echo) image, all pooled, no conditioning.
Variable slice counts (22-52) are handled per file.
Per-image normalization by max magnitude.

Usage:
    python build_shard.py /path/to/mat_dir /path/to/out_dir
"""

import sys
import os
import glob
import json
import numpy as np

IMG_H = 256
IMG_W = 256
N_ECHOES = 6  # confirmed from data; script will assert this per file


# ---------- readers ----------

def read_images_h5(path):
    """Read the complex 'images' array from a v7.3 (HDF5) .mat file.
    Returns a complex numpy array in MATLAB order (H, W, slices, echoes)."""
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

        raw = dset[()]  # HDF5 order, reversed dims: (echoes, slices, W, H)
        # reconstruct complex from compound real/imag fields
        if raw.dtype.names and "real" in raw.dtype.names:
            arr = raw["real"] + 1j * raw["imag"]
        else:
            arr = np.asarray(raw)
        # reverse dim order back to MATLAB layout (H, W, slices, echoes)
        arr = np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))
    return arr


def read_images_scipy(path):
    """Read the complex 'images' array from a pre-v7.3 .mat file."""
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    for k in d.keys():
        if k.startswith("__"):
            continue
        obj = d[k]
        if hasattr(obj, "_fieldnames") and "images" in obj._fieldnames:
            return np.asarray(obj.images)  # already (H, W, slices, echoes)
    raise KeyError(f"no 'images' field found in {path}")


def read_images(path):
    """Try HDF5 first (v7.3), fall back to scipy."""
    try:
        return read_images_h5(path)
    except Exception:
        return read_images_scipy(path)


# ---------- conversion ----------

def count_samples(files):
    """Pass 1: total N = sum over files of (slices * echoes)."""
    total = 0
    per_file = []
    for p in files:
        arr = read_images(p)
        assert arr.ndim == 4, f"{p}: expected 4D images, got {arr.shape}"
        h, w, n_sl, n_ec = arr.shape
        assert (h, w) == (IMG_H, IMG_W), \
            f"{p}: matrix {h}x{w} != {IMG_H}x{IMG_W} (crop/pad needed)"
        assert n_ec == N_ECHOES, f"{p}: {n_ec} echoes != {N_ECHOES}"
        n = n_sl * n_ec
        per_file.append({"file": os.path.basename(p),
                         "slices": int(n_sl), "echoes": int(n_ec), "n": int(n)})
        total += n
        print(f"  {os.path.basename(p)}: {n_sl} slices x {n_ec} echoes = {n}")
    return total, per_file


def build(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dat_path = os.path.join(out_dir, "fam3t_slices.dat")
    json_path = os.path.join(out_dir, "fam3t_slices.json")

    print("Pass 1: counting samples...")
    total_n, per_file = count_samples(files)
    print(f"Total samples N = {total_n}")

    print(f"Pass 2: writing memmap {dat_path}  shape ({total_n}, 2, {IMG_H}, {IMG_W})")
    mm = np.memmap(dat_path, dtype=np.float32, mode="w+",
                   shape=(total_n, 2, IMG_H, IMG_W))

    idx = 0
    for p in files:
        arr = read_images(p)                     # (H, W, slices, echoes) complex
        h, w, n_sl, n_ec = arr.shape
        for s in range(n_sl):
            for e in range(n_ec):
                img = arr[:, :, s, e]            # complex 256x256
                mag = np.abs(img).max()
                if mag <= 0 or not np.isfinite(mag):
                    mag = 1.0                    # guard empty/NaN slices
                img = img / mag                  # per-image normalization
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
    if len(sys.argv) != 3:
        print("usage: python build_shard.py /path/to/mat_dir /path/to/out_dir")
        sys.exit(1)
    mat_dir, out_dir = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    if not files:
        print(f"no .mat files found in {mat_dir}")
        sys.exit(1)
    print(f"found {len(files)} .mat files")
    build(files, out_dir)


if __name__ == "__main__":
    main()