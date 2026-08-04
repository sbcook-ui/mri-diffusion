import numpy as np, json, matplotlib.pyplot as plt

meta = json.load(open("data/shards/fam3t_slices.json"))
N = meta["n"]
mm = np.memmap("data/shards/fam3t_slices.dat", dtype=np.float32,
               mode="r", shape=tuple(meta["shape"]))

print(f"N={N}, shape={mm.shape}, dtype={mm.dtype}")
print(f"value range: min={mm.min():.3f} max={mm.max():.3f}")

i = N // 2                      # a middle sample
real, imag = mm[i, 0], mm[i, 1]
mag = np.sqrt(real**2 + imag**2)

plt.imshow(mag, cmap="gray")
plt.title(f"sample {i} (magnitude)")
plt.axis("off")
plt.savefig("data/shards/check.png", dpi=100, bbox_inches="tight")
print("saved data/shards/check.png")