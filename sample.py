"""
sample.py

Generate MRI slices from a trained checkpoint using standard DDPM ancestral sampling.
Loads the ADM UNetModel, samples 2-channel complex, saves magnitude PNGs AND the
raw complex arrays (.npy) so downstream analysis uses full precision, not 8-bit PNGs.

    python sample.py --ckpt checkpoints/model_000200.pt --n 100

Outputs to --out_dir:
    sample_{i}.png       magnitude preview (8-bit, for eyeballing)
    generated_raw.npy    all samples as one array, shape (n, 2, H, W), float32
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

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
def sample(model, diff_T, image_size, n, device, batch=8):
    """Standard DDPM ancestral sampling from pure noise, in batches to fit memory."""
    betas = torch.linspace(1e-4, 0.02, diff_T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    out = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        x = torch.randn(b, 2, image_size, image_size, device=device)
        for t in reversed(range(diff_T)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            a = alphas[t]
            ab = abar[t]
            coef = (1 - a) / (1 - ab).sqrt()
            mean = (x - coef * eps) / a.sqrt()
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + betas[t].sqrt() * noise
            else:
                x = mean
        out.append(x.cpu())
        done += b
        print(f"  generated {done}/{n}", flush=True)
    return torch.cat(out, dim=0)  # (n, 2, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out_dir", default="samples")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    print(f"loaded {args.ckpt}  step={ck['step']}  cfg={cfg}")

    model = build_unet(cfg["image_size"], 2, cfg["num_channels"],
                       cfg["num_res_blocks"], cfg["attn_res"],
                       cfg.get("num_head_channels", -1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    imgs = sample(model, cfg["timesteps"], cfg["image_size"], args.n, device,
                  batch=args.batch)
    imgs_np = imgs.numpy().astype(np.float32)  # (n, 2, H, W)

    os.makedirs(args.out_dir, exist_ok=True)

    # save raw complex (as 2-channel real/imag) for full-precision analysis
    np.save(os.path.join(args.out_dir, "generated_raw.npy"), imgs_np)

    # save magnitude PNG previews
    for i in range(args.n):
        real, imag = imgs_np[i, 0], imgs_np[i, 1]
        mag = np.sqrt(real**2 + imag**2)
        plt.imsave(os.path.join(args.out_dir, f"sample_{i}.png"), mag, cmap="gray")

    print(f"saved {args.n} samples + generated_raw.npy to {args.out_dir}/")


if __name__ == "__main__":
    main()