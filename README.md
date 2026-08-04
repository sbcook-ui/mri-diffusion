# Generating Liver MRI with a Diffusion Model

Sam Cook, Xiao Shi, Jiayi Tang, Diego Hernando, Ulugbek Kamilov

UW-Madison ECE Summer Program for Advanced Research and Knowledge (SPARK), Summer 2026

<!-- Optional cover figure. Put an image (e.g. a real scan next to a generated slice) in a figures/ folder and uncomment. -->
<!-- ![cover](figures/cover.png) -->

## Overview

This project trains a diffusion model to generate realistic, anatomically plausible liver MRI slices, learned from multi-echo clinical data rather than reproduced as memorized copies of real scans. Public ML-ready datasets of multi-echo liver MRI are scarce, and standard augmentation (flips, crops) only reshuffles existing scans without adding new information. A diffusion model instead learns the underlying data distribution, so it can generate genuinely new scans. These generated images can expand training data and act as a learned prior that guides reconstruction and denoising toward realistic results.

The work produced two main findings:

1. **Realism.** The model generates recognizable liver anatomy, though not yet at real-scan image quality, as measured by FID (lower is better). Generated samples reached an FID of 34.6, compared to 27.0 for a different real sequence (IDEAL IQ) and 13.8 for real-vs-real.
2. **No memorization.** A nearest-neighbor similarity test found no evidence that the model copies its training data. Generated slices are less similar to the training set (SSIM 0.84 max / 0.67 median) than independent real sequences are (0.91 / 0.70), indicating the model produces new images rather than reproductions.

Next step: denoise real scans by adding known noise and running the reverse diffusion process to recover a cleaner image.

## Environment setting

### 1) Clone the repository

```
git clone https://github.com/<username>/liver-mri-diffusion
cd liver-mri-diffusion
```

### 2) Data

The model is trained on a clinical liver MRI dataset from UW-Health (~22,000 complex-valued multi-echo slices). This dataset is not publicly available.

<!-- Add a line here on how lab members can access the data, or note it as restricted / IRB-controlled. -->

### 3) Virtual environment setup

```
conda create -n liver-mri-diffusion python=3.9
conda activate liver-mri-diffusion
pip install -r requirements.txt
```

<!-- Generate requirements.txt from your active env with:  pip freeze > requirements.txt
     then trim anything you didn't actually use. -->

## Running the code

<!-- CONFIRM THESE COMMANDS. Fill in the exact flags your scripts expect
     (paste your argparse block or a working command and I'll finalize this). -->

Preprocess the raw data into training shards:
```
python preprocessing/build_shard.py [args]
```

Train the diffusion model:
```
python train.py [args]
```

Generate liver slices from the trained model:
```
python sample.py [args]
```

Evaluate memorization (nearest-neighbor similarity to real training scans):
```
python memorization_check.py [args]
python memorization_check_psnr.py [args]
```

Prepare images and compute FID:
```
python fid_prep.py [args]
```

<!-- Cluster note: the paired .sh / .sub files are HTCondor job scripts for running
     the above on the compute cluster. Briefly say which .sub runs which job if useful. -->

## Implementation detail

<!-- Verify these descriptions match what each file actually does. -->

```
train.py                        # Trains the U-Net diffusion model
sample.py                       # Generates slices from a trained model
adm_unet.py                     # U-Net (ADM-style) architecture / denoising backbone
capture_diffusion.py            # Captures the reverse diffusion process (t=999 -> t=0)
memorization_check.py           # Nearest-neighbor memorization test (SSIM)
memorization_check_psnr.py      # Nearest-neighbor memorization test (PSNR)
fid_prep.py                     # Prepares generated/real images for FID computation
│
├── preprocessing/              # Builds training data shards from raw scans
├── configs/                    # Experiment configuration files
└── scripts/                    # Helper / job scripts
```

## Code references

We build on the [Measurement Score-based diffusion Model (MSM)](https://arxiv.org/abs/2505.11853) framework, which in turn adapts [guided-diffusion](https://github.com/openai/guided-diffusion).

## Data reference

Heidenreich JF, Tang J, Tamada D, Müller L, Grunz JP, do Vale Souza R, Anagnostopoulos A, Pirasteh A, Reeder SB, Hernando D. Motion-Insensitive Flip Angle Modulated Liver Proton Density Fat-Fraction and R2* Mapping During Free-Breathing MRI in a Clinical Setting. *J Magn Reson Imaging*. 2025 Nov;62(5):1452-1463. doi:10.1002/jmri.70033