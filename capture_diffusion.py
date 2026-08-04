"""
capture_diffusion.py

Runs your exact DDPM ancestral sampler (same math as sample.py) for ONE image and
snapshots the running image x_t at chosen timesteps, so we can show the real
reverse-diffusion process: pure noise -> partially denoised -> final.

Saves raw complex snapshots (full precision) + the timestep of each, for rendering.

    python capture_diffusion.py --ckpt checkpoints/model_000200.pt --seed 0

Output (to --out_dir):
    diffusion_steps.npy   shape (k, 2, H, W) float32  -- one slice per captured t
    diffusion_meta.json   {"timesteps_T": T, "captured_t": [...], "seed": ...}
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import numpy as np
import torch

from adm_unet import UNetModel


def build_unet(image_size, channels, num_channels, num_res_blocks, attn_res,
               num_head_channels):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image_size {image_size}")
    attention_ds = tuple(image_size // int(r) for r in attn_res)
    return UNetModel(
        image_size=image_size, in_channels=channels, model_channels=num_channels,
        out_channels=channels, num_res_blocks=num_res_blocks,
        attention_resolutions=attention_ds, dropout=0.0, channel_mult=channel_mult,
        num_classes=None, use_checkpoint=False, use_fp16=False,
        num_heads=4, num_head_channels=num_head_channels, use_scale_shift_norm=True,
        resblock_updown=True,
    )


@torch.no_grad()
def sample_with_capture(model, diff_T, image_size, device, capture_t, seed=0):
    """One-image DDPM ancestral sampling; record x at each timestep in capture_t.
    capture_t are 'about to denoise step t' snapshots (x_t, the state at time t)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 2, image_size, image_size, device=device, generator=g)

    betas = torch.linspace(1e-4, 0.02, diff_T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    capture_set = set(int(t) for t in capture_t)
    snaps, snap_t = [], []

    for t in reversed(range(diff_T)):
        if t in capture_set:                      # snapshot BEFORE denoising step t
            snaps.append(x.squeeze(0).cpu().numpy().astype(np.float32))
            snap_t.append(int(t))
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        a, ab = alphas[t], abar[t]
        coef = (1 - a) / (1 - ab).sqrt()
        mean = (x - coef * eps) / a.sqrt()
        if t > 0:
            noise = torch.randn(x.shape, device=device, generator=g)
            x = mean + betas[t].sqrt() * noise
        else:
            x = mean

    # always capture the final image (t effectively 0, fully denoised)
    snaps.append(x.squeeze(0).cpu().numpy().astype(np.float32))
    snap_t.append(0)
    return np.stack(snaps), snap_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="diffusion_capture")
    p.add_argument("--seed", type=int, default=0)
    # extra intermediate steps included so you can choose the best middle panel
    p.add_argument("--capture", default="",
                   help="comma-separated timesteps to snapshot; blank = auto from T")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    T = cfg["timesteps"]
    print(f"loaded {args.ckpt} step={ck['step']} T={T} image_size={cfg['image_size']}")

    model = build_unet(cfg["image_size"], 2, cfg["num_channels"],
                       cfg["num_res_blocks"], cfg["attn_res"],
                       cfg.get("num_head_channels", -1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.capture.strip():
        capture_t = [int(s) for s in args.capture.split(",")]
    else:
        # pure noise (T-1), then a spread of intermediates to pick a middle from
        capture_t = [T - 1,
                     int(0.75 * T), int(0.6 * T), int(0.5 * T),
                     int(0.4 * T), int(0.25 * T), int(0.1 * T)]
    print(f"capturing t = {capture_t} (+ final t=0)")

    snaps, snap_t = sample_with_capture(model, T, cfg["image_size"], device,
                                        capture_t, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "diffusion_steps.npy"), snaps)
    with open(os.path.join(args.out_dir, "diffusion_meta.json"), "w") as f:
        json.dump({"timesteps_T": int(T), "captured_t": snap_t, "seed": args.seed}, f)
    print(f"saved {snaps.shape[0]} snapshots (t={snap_t}) to {args.out_dir}/")


if __name__ == "__main__":
    main()