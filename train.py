"""
train.py

Standard DDPM training around the ADM UNetModel from the MSM repo.
Image-domain, complex-as-2-channels. No k-space, no coils, no measurement code.

Smoke mode (default): tiny model, 64x64, few steps, runs on CPU in minutes.
    python train.py --smoke

Full mode (for CHTC/GPU later): full ADM width, 256x256.
    python train.py --full

Outputs checkpoints to ../checkpoints/.
"""

# --- OpenMP duplicate-runtime fix (Windows/conda). Must be set before torch. ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import math
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# import the self-contained ADM U-Net (lifted from the repo, k-space stripped)
from adm_unet import UNetModel


# ----------------------------- data -----------------------------

class ShardDataset(torch.utils.data.Dataset):
    """Reads the memmap shard (N, 2, 256, 256) float32. Optionally downsamples."""
    def __init__(self, shard_dir, image_size):
        meta = json.load(open(os.path.join(shard_dir, "fam3t_slices.json")))
        self.shape = tuple(meta["shape"])
        self.n = meta["n"]
        self.mm = np.memmap(os.path.join(shard_dir, "fam3t_slices.dat"),
                            dtype=np.float32, mode="r", shape=self.shape)
        self.image_size = image_size
        self.native = self.shape[-1]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.mm[i]))  # (2, H, W)
        if self.image_size != self.native:
            x = F.interpolate(x[None], size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)[0]
        return x


# ----------------------------- diffusion (DDPM) -----------------------------

class Diffusion:
    """Minimal DDPM: linear beta schedule, epsilon-prediction MSE loss."""
    def __init__(self, timesteps, device):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        self.abar = torch.cumprod(alphas, dim=0)      # alpha_bar_t
        self.device = device

    def q_sample(self, x0, t, noise):
        """Add noise to x0 at timestep t: x_t = sqrt(abar) x0 + sqrt(1-abar) noise."""
        a = self.abar[t].sqrt().view(-1, 1, 1, 1)
        b = (1 - self.abar[t]).sqrt().view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)          # ADM U-Net predicts the noise (epsilon)
        return F.mse_loss(pred, noise)


# ----------------------------- model builder -----------------------------

def build_unet(image_size, channels, num_channels, num_res_blocks, attn_res,
               num_head_channels):
    """Construct the ADM UNetModel for 2-channel complex, epsilon-prediction."""
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image_size {image_size}")
    attention_ds = tuple(image_size // int(r) for r in attn_res)
    return UNetModel(
        image_size=image_size,
        in_channels=channels,
        model_channels=num_channels,
        out_channels=channels,          # epsilon has same channels as input
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_ds,
        dropout=0.0,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=num_head_channels,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )


# ----------------------------- train loop -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny CPU smoke test")
    p.add_argument("--gputest", action="store_true",
                   help="full-size model, few steps: validates GPU path")
    p.add_argument("--run100k", action="store_true",
                   help="full-size model, 100k steps, frequent checkpoints")
    p.add_argument("--run300k", action="store_true",
                   help="full-size model, 300k steps, checkpoint every 50k")
    p.add_argument("--full", action="store_true", help="full ADM (GPU/CHTC)")
    p.add_argument("--shard_dir", default="data/shards")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--resume", default="", help="path to checkpoint to resume from")
    args = p.parse_args()

    if not (args.smoke or args.gputest or args.run100k or args.run300k or args.full):
        args.smoke = True  # default to smoke

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        mode = "smoke"
        # tiny model -> num_head_channels=-1 (uses num_heads directly)
        cfg = dict(image_size=64, num_channels=32, num_res_blocks=1,
                   attn_res=["16"], num_head_channels=-1, timesteps=100,
                   batch_size=4, steps=200, log_every=20, save_every=200)
    elif args.gputest:
        mode = "gputest"
        # full-size architecture, but only a few hundred steps
        cfg = dict(image_size=256, num_channels=128, num_res_blocks=2,
                   attn_res=["32", "16", "8"], num_head_channels=64,
                   timesteps=1000, batch_size=8, steps=300,
                   log_every=25, save_every=300)
    elif args.run100k:
        mode = "run100k"
        # full-size, 100k steps, checkpoint every 10k so progress is never lost.
        # batch_size=8 needs a large GPU (>=40GB); submit file enforces this.
        cfg = dict(image_size=256, num_channels=128, num_res_blocks=2,
                   attn_res=["32", "16", "8"], num_head_channels=64,
                   timesteps=1000, batch_size=8, steps=100_000,
                   log_every=500, save_every=10_000)
    elif args.run300k:
        mode = "run300k"
        # full-size, 300k steps, checkpoint every 50k (6 checkpoints total)
        cfg = dict(image_size=256, num_channels=128, num_res_blocks=2,
                   attn_res=["32", "16", "8"], num_head_channels=64,
                   timesteps=1000, batch_size=8, steps=300_000,
                   log_every=500, save_every=50_000)
    else:  # full
        mode = "full"
        cfg = dict(image_size=256, num_channels=128, num_res_blocks=2,
                   attn_res=["32", "16", "8"], num_head_channels=64,
                   timesteps=1000, batch_size=8, steps=1_000_000,
                   log_every=1000, save_every=50000)

    print(f"device={device}  mode={mode}")
    print(f"config: {cfg}")

    ds = ShardDataset(args.shard_dir, cfg["image_size"])
    print(f"dataset N={len(ds)}  native={ds.native}  train_size={cfg['image_size']}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    model = build_unet(cfg["image_size"], 2, cfg["num_channels"],
                       cfg["num_res_blocks"], cfg["attn_res"],
                       cfg["num_head_channels"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    diff = Diffusion(cfg["timesteps"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # optional resume: restore model, optimizer, and step count
    start_step = 0
    if args.resume:
        print(f"resuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"resumed at step {start_step}")

    model.train()
    step = start_step
    data_iter = iter(loader)
    while step < cfg["steps"]:
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        x0 = x0.to(device)

        opt.zero_grad()
        loss = diff.loss(model, x0)
        loss.backward()
        opt.step()

        if step % cfg["log_every"] == 0:
            print(f"step {step:>7}  loss {loss.item():.4f}", flush=True)
        if step > 0 and step % cfg["save_every"] == 0:
            path = os.path.join(args.ckpt_dir, f"model_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "cfg": cfg, "step": step}, path)
            print(f"saved {path}", flush=True)
        step += 1

    path = os.path.join(args.ckpt_dir, f"model_{step:06d}.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "step": step}, path)
    print(f"saved final {path}")
    print("done.")


if __name__ == "__main__":
    main()