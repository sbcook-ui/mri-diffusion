"""
inspect_mat.py

Run this on ONE FAM 3T .mat file first, before bulk conversion.
Confirms: the struct variable name, the 'images' shape, dim order,
complex handling, and slice/echo counts.

Usage:
    python inspect_mat.py /path/to/one_scan.mat
"""

import sys
import numpy as np


def try_h5py(path):
    import h5py
    print(f"[h5py] opening {path}")
    with h5py.File(path, "r") as f:
        top = [k for k in f.keys() if not k.startswith("#")]
        print(f"  top-level keys: {top}")

        # Find the group/struct that contains an 'images' field.
        for k in top:
            obj = f[k]
            if isinstance(obj, h5py.Group) and "images" in obj:
                print(f"  -> struct variable name: '{k}'")
                img = obj["images"]
                print(f"     images raw shape (HDF5 order): {img.shape}")
                print(f"     images dtype: {img.dtype}")
                # complex is a compound dtype with fields 'real','imag'
                if img.dtype.names:
                    print(f"     compound fields: {img.dtype.names}  "
                          f"(complex stored as real/imag)")
                # read a tiny corner to confirm we can reconstruct complex
                sample = img[0, 0] if img.ndim >= 2 else img[()]
                print(f"     sample element type: {type(sample)}")
                # report the other fields for sanity
                for fld in obj.keys():
                    try:
                        val = obj[fld]
                        print(f"     field '{fld}': shape {val.shape}, dtype {val.dtype}")
                    except Exception:
                        pass
                return True
        # images might sit at top level (not inside a struct group)
        if "images" in f:
            img = f["images"]
            print(f"  -> 'images' at top level, shape {img.shape}, dtype {img.dtype}")
            return True
    print("  could not locate an 'images' field via h5py.")
    return False


def try_scipy(path):
    from scipy.io import loadmat
    print(f"[scipy] opening {path} (pre-v7.3 path)")
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in d.keys() if not k.startswith("__")]
    print(f"  top-level keys: {keys}")
    for k in keys:
        obj = d[k]
        if hasattr(obj, "_fieldnames") and "images" in obj._fieldnames:
            print(f"  -> struct variable name: '{k}'")
            img = obj.images
            print(f"     images shape (MATLAB order preserved): {img.shape}")
            print(f"     images dtype: {img.dtype} (complex if 'complex' in dtype)")
            return True
    print("  could not locate an 'images' field via scipy.")
    return False


def main():
    if len(sys.argv) != 2:
        print("usage: python inspect_mat.py /path/to/one_scan.mat")
        sys.exit(1)
    path = sys.argv[1]

    # v7.3 files are HDF5; older files are not. Try h5py first, fall back.
    try:
        ok = try_h5py(path)
        if ok:
            print("\nThis is a v7.3 (HDF5) file. Bulk converter will use h5py.")
            print("NOTE: HDF5 reverses dims -> your 256x256xSLICESx6 reads as "
                  "(6, SLICES, 256, 256). The converter transposes back.")
            return
    except Exception as e:
        print(f"  h5py failed ({e}); trying scipy...")

    try:
        ok = try_scipy(path)
        if ok:
            print("\nThis is a pre-v7.3 file. Bulk converter will use scipy.io.loadmat.")
            return
    except Exception as e:
        print(f"  scipy failed ({e}).")

    print("\nCould not read the file with either reader. Check the path/format.")


if __name__ == "__main__":
    main()