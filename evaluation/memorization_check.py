"""
memorization_check.py

Memorization check for the diffusion model. For each generated image, finds the
closest real training slices by PSNR and by SSIM. If a generated image is nearly
identical to a specific real slice (very high SSIM/PSNR), that is evidence the
model memorized training data rather than learning to generate novel samples.

Compares on MAGNITUDE images (standard for PSNR/SSIM on complex MRI).
Both generated and real slices are per-image min-max normalized to [0,1] before
comparison so scores are not dominated by overall brightness.

Inputs:
    --generated   path to generated_raw.npy   (n, 2, H, W) float32  (real,imag)
    --shard       path to fam3t_slices.dat     memmap (N, 2, H, W) float32
    --shard_meta  path to fam3t_slices.json
    --topk        how many nearest real slices to keep per generated image (default 3)

Outputs (to --out_dir):
    memorization_scores.csv    one row per (generated image, rank, metric) with
                               the matched real-slice index and PSNR/SSIM
    montage_{i}.png            generated image next to its topk nearest real slices
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def to_magnitude(arr2):
    """arr2: (..., 2, H, W) -> magnitude (..., H, W)."""
    return torch.sqrt(arr2[..., 0, :, :] ** 2 + arr2[..., 1, :, :] ** 2)


def per_image_minmax(x, eps=1e-8):
    """Normalize each image to [0,1]. x: (B, H, W)."""
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - mn) / (mx - mn + eps)
    return flat.view_as(x)


def psnr_batch(gen, reals, eps=1e-8):
    """
    gen:   (H, W)              one generated image, normalized [0,1]
    reals: (B, H, W)           batch of real slices, normalized [0,1]
    returns PSNR (B,) in dB. Max value is 1.0 since normalized.
    """
    mse = ((reals - gen.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)   # (1,1,size,size)


def ssim_batch(gen, reals, window=None, C1=0.01**2, C2=0.03**2):
    """
    Windowed (Gaussian) SSIM between one generated image and a batch of reals.
    gen:   (H, W)     reals: (B, H, W), both normalized [0,1].
    Returns mean SSIM (B,) over the local SSIM map for each real slice.
    Uses an 11x11 Gaussian window (sigma 1.5), the canonical Wang et al. setup,
    with 'valid' convolution so border pixels don't bias the score.
    """
    if window is None:
        window = _gaussian_window(11, 1.5, reals.device)

    x = reals.unsqueeze(1)                 # (B,1,H,W)
    y = gen.view(1, 1, *gen.shape)         # (1,1,H,W) -> broadcasts over batch

    mu_x = F.conv2d(x, window)             # padding=0 (valid)
    mu_y = F.conv2d(y, window)             # (1,1,h,w), reused across batch

    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window) - mu_x2
    sigma_y2 = F.conv2d(y * y, window) - mu_y2
    sigma_xy = F.conv2d(x * y, window) - mu_xy

    ssim_map = (((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
                ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
    return ssim_map.mean(dim=(1, 2, 3))    # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--shard_meta", required=True)
    ap.add_argument("--out_dir", default="memorization_out")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--real_batch", type=int, default=512,
                    help="how many real slices to score at once")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load generated images -> magnitude, normalized ----
    gen_raw = np.load(args.generated)          # (n, 2, H, W)
    n_gen = gen_raw.shape[0]
    H, W = gen_raw.shape[2], gen_raw.shape[3]
    gen_t = torch.from_numpy(gen_raw).to(device)
    gen_mag = to_magnitude(gen_t)              # (n, H, W)
    gen_mag = per_image_minmax(gen_mag)        # (n, H, W)
    print(f"generated: {n_gen} images, {H}x{W}")

    # ---- open the real-slice shard as a memmap ----
    meta = json.load(open(args.shard_meta))
    shape = tuple(meta["shape"])               # (N, 2, H, W)
    N = shape[0]
    reals_mm = np.memmap(args.shard, dtype=np.float32, mode="r", shape=shape)
    print(f"real slices: {N}")

    # best[k] per generated image: track topk by PSNR and by SSIM separately
    # store as lists of (score, real_index)
    best_psnr = [[] for _ in range(n_gen)]
    best_ssim = [[] for _ in range(n_gen)]

    def update_topk(store, scores, idxs, k):
        # store: list of (score, idx); keep k highest
        for sc, ix in zip(scores, idxs):
            store.append((float(sc), int(ix)))
        store.sort(key=lambda t: t[0], reverse=True)
        del store[k:]

    # ---- stream through the real slices in batches ----
    rb = args.real_batch
    for start in range(0, N, rb):
        end = min(start + rb, N)
        chunk = np.asarray(reals_mm[start:end])          # (b,2,H,W)
        chunk_t = torch.from_numpy(chunk).to(device)
        real_mag = to_magnitude(chunk_t)                 # (b,H,W)
        real_mag = per_image_minmax(real_mag)            # (b,H,W)
        idxs = list(range(start, end))

        for gi in range(n_gen):
            ps = psnr_batch(gen_mag[gi], real_mag).cpu().numpy()
            ss = ssim_batch(gen_mag[gi], real_mag).cpu().numpy()
            update_topk(best_psnr[gi], ps, idxs, args.topk)
            update_topk(best_ssim[gi], ss, idxs, args.topk)

        if (start // rb) % 5 == 0:
            print(f"  scored real slices {end}/{N}", flush=True)

    # ---- write CSV ----
    csv_path = os.path.join(args.out_dir, "memorization_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generated_index", "metric", "rank",
                    "real_slice_index", "score"])
        for gi in range(n_gen):
            for rank, (sc, ix) in enumerate(best_psnr[gi], 1):
                w.writerow([gi, "PSNR", rank, ix, f"{sc:.4f}"])
            for rank, (sc, ix) in enumerate(best_ssim[gi], 1):
                w.writerow([gi, "SSIM", rank, ix, f"{sc:.4f}"])
    print(f"wrote {csv_path}")

    # ---- montages: generated + its topk SSIM-nearest reals ----
    def mag_from_mm(ix):
        a = np.asarray(reals_mm[ix])
        m = np.sqrt(a[0] ** 2 + a[1] ** 2)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m

    for gi in range(n_gen):
        cols = 1 + args.topk
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
        g = gen_mag[gi].cpu().numpy()
        axes[0].imshow(g, cmap="gray"); axes[0].set_title(f"generated {gi}")
        axes[0].axis("off")
        for r, (sc, ix) in enumerate(best_ssim[gi]):
            axes[r + 1].imshow(mag_from_mm(ix), cmap="gray")
            axes[r + 1].set_title(f"real {ix}\nSSIM {sc:.3f}")
            axes[r + 1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"montage_{gi:03d}.png"),
                    dpi=90, bbox_inches="tight")
        plt.close()

    # ---- quick summary: the most-suspicious matches ----
    top_overall = []
    for gi in range(n_gen):
        if best_ssim[gi]:
            top_overall.append((best_ssim[gi][0][0], gi, best_ssim[gi][0][1]))
    top_overall.sort(reverse=True)
    print("\nMost similar generated->real matches (by SSIM):")
    for sc, gi, ix in top_overall[:10]:
        print(f"  generated {gi:3d}  ->  real {ix:6d}   SSIM {sc:.4f}")
    print(f"\nMontages + CSV in {args.out_dir}/")


if __name__ == "__main__":
    main()