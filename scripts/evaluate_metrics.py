"""Quantitative Evaluation Script for MineArt Diffusion: FID and KID.

Computes Fréchet Inception Distance (FID) and Kernel Inception Distance (KID)
between real Minecraft dataset images and generated artwork samples using Inception-v3.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import scipy.linalg
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

from diffusion import MineArtUNet, GaussianDiffusionEngine, PromptTokenizer, PAINTING_RATIOS


class InceptionFeatureExtractor(nn.Module):
    """Inception-v3 feature extractor extracting 2048-d representations."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.fc = nn.Identity()
        inception.eval()

        self.inception = inception.to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: tensor in [0, 1] range of shape (B, 3, H, W)."""
        x_norm = self.preprocess(x)
        return self.inception(x_norm)


def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Calculates Fréchet Inception Distance between two Gaussian distributions."""
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = None, coef0: float = 1.0) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * np.dot(x, y.T) + coef0) ** degree


def compute_kid(features_real: np.ndarray, features_gen: np.ndarray, num_subsets: int = 50, subset_size: int = 50) -> Tuple[float, float]:
    """Calculates Kernel Inception Distance (KID) with mean and standard deviation."""
    n_real = len(features_real)
    n_gen = len(features_gen)
    actual_sub_size = min(subset_size, n_real, n_gen)

    mmd_list = []
    rng = np.random.default_rng(42)

    for _ in range(num_subsets):
        idx_r = rng.choice(n_real, actual_sub_size, replace=False)
        idx_g = rng.choice(n_gen, actual_sub_size, replace=False)

        xr = features_real[idx_r]
        xg = features_gen[idx_g]

        k_rr = polynomial_kernel(xr, xr)
        k_gg = polynomial_kernel(xg, xg)
        k_rg = polynomial_kernel(xr, xg)

        np.fill_diagonal(k_rr, 0)
        np.fill_diagonal(k_gg, 0)

        m = actual_sub_size
        mmd = (k_rr.sum() / (m * (m - 1))) + (k_gg.sum() / (m * (m - 1))) - (2 * k_rg.sum() / (m * m))
        mmd_list.append(mmd)

    return float(np.mean(mmd_list)), float(np.std(mmd_list))


def extract_real_features(
    image_dir: str,
    extractor: InceptionFeatureExtractor,
    num_samples: int = 100,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Extracts features from random real dataset images."""
    img_dir = Path(image_dir)
    all_imgs = [p for p in img_dir.glob("*.png")] + [p for p in img_dir.glob("*.jpg")]
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {image_dir}")

    selected = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    print(f"[*] Extracting Inception features from {len(selected)} real Minecraft images...")

    to_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    features_list = []
    for i in range(0, len(selected), batch_size):
        batch_paths = selected[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    tensors.append(to_tensor(img.convert("RGB")))
            except Exception:
                continue
        if tensors:
            batch = torch.stack(tensors).to(device)
            feat = extractor(batch)
            features_list.append(feat.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def extract_generated_features(
    model: MineArtUNet,
    diffusion: GaussianDiffusionEngine,
    tokenizer: PromptTokenizer,
    extractor: InceptionFeatureExtractor,
    prompts: List[str],
    num_samples: int = 100,
    batch_size: int = 16,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
    resolution: Tuple[int, int] = (64, 64),
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, List[Image.Image]]:
    """Synthesizes images and extracts Inception features."""
    print(f"[*] Generating {num_samples} artwork samples via DDIM ({ddim_steps} steps, CFG {guidance_scale})...")
    features_list = []
    sample_images = []

    model.eval()
    batches = (num_samples + batch_size - 1) // batch_size

    for b in tqdm(range(batches), desc="Sampling & Evaluating"):
        cur_batch = min(batch_size, num_samples - b * batch_size)
        batch_prompts = [random.choice(prompts) for _ in range(cur_batch)]

        prompt_tokens = torch.stack([tokenizer.encode(p, max_length=16) for p in batch_prompts]).to(device)
        uncond_tokens = tokenizer.null_tokens(max_length=16).repeat(cur_batch, 1).to(device)

        with torch.no_grad():
            sampled = diffusion.sample_text_to_image(
                model=model,
                prompt_tokens=prompt_tokens,
                uncond_tokens=uncond_tokens,
                shape=(cur_batch, 3, resolution[1], resolution[0]),
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
            )
            feat = extractor(sampled)
            features_list.append(feat.cpu().numpy())

            if len(sample_images) < 10:
                for img_t in sampled:
                    img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sample_images.append(Image.fromarray(img_np))
                    if len(sample_images) >= 10:
                        break

    return np.concatenate(features_list, axis=0), sample_images


def main():
    parser = argparse.ArgumentParser(description="MineArt Diffusion FID / KID Evaluation Engine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_diffusion.pt", help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/diffusion_tokenizer.json", help="Tokenizer path")
    parser.add_argument("--image-dir", type=str, default="data/processed_64x64", help="Ground truth real images directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of real & generated samples to evaluate")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--guidance", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--size", type=int, default=64, help="Resolution (64 for 64x64)")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--test-untrained-baseline", action="store_true", help="Also compute FID for untrained random noise baseline")
    parser.add_argument("--output-json", type=str, default="checkpoints/evaluation_results.json", help="Output JSON results")
    parser.add_argument("--save-samples-dir", type=str, default="output_samples/eval_samples", help="Directory to save sample images")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[!] Running Evaluation on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    extractor = InceptionFeatureExtractor(device=device)

    # Fall back to data/images if data/processed_64x64 is not populated
    img_dir_path = Path(args.image_dir)
    if not img_dir_path.exists() or len(list(img_dir_path.glob("*.png"))) == 0:
        args.image_dir = "data/images"

    # 1. Extract Real Features
    real_features = extract_real_features(
        image_dir=args.image_dir,
        extractor=extractor,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    results = {
        "num_samples": args.num_samples,
        "ddim_steps": args.steps,
        "guidance_scale": args.guidance,
        "resolution": f"{args.size}x{args.size}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 2. (Optional) Evaluate Untrained Baseline
    if args.test_untrained_baseline:
        print("\n--- Evaluating Level 0 Baseline (Untrained Model / Random Gaussian Noise) ---")
        noise_batch = torch.rand((args.num_samples, 3, args.size, args.size), device=device)
        with torch.no_grad():
            noise_feats = extractor(noise_batch).cpu().numpy()
        mu_noise = np.mean(noise_feats, axis=0)
        sigma_noise = np.cov(noise_feats, rowvar=False)
        baseline_fid = compute_fid(mu_real, sigma_real, mu_noise, sigma_noise)
        baseline_kid_mean, baseline_kid_std = compute_kid(real_features, noise_feats)
        print(f"[BASELINE] Pure Noise FID: {baseline_fid:.2f} (Expected > 300)")
        print(f"[BASELINE] Pure Noise KID: {baseline_kid_mean:.4f} ± {baseline_kid_std:.4f}")
        results["baseline_untrained"] = {
            "fid": baseline_fid,
            "kid_mean": baseline_kid_mean,
            "kid_std": baseline_kid_std,
        }

    # 3. Evaluate Trained Model Checkpoint
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        print(f"[!] Checkpoint not found at '{checkpoint_path}'. Skipping trained model evaluation.")
        return

    tokenizer = PromptTokenizer.load(str(tokenizer_path))
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    model = MineArtUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        vocab_size=len(tokenizer.vocab),
        text_embed_dim=128,
        time_embed_dim=128,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusionEngine(timesteps=1000, device=device)

    sample_prompts = [
        "minecraft plains landscape", "sunflower plains daylight", "cherry blossom forest",
        "desert temple sand dunes", "ocean coral reef water", "creeper dark oak forest",
        "nether fortress lava blocks", "ancient city warden stone", "villager house mountains"
    ]

    gen_features, sample_images = extract_generated_features(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        extractor=extractor,
        prompts=sample_prompts,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ddim_steps=args.steps,
        guidance_scale=args.guidance,
        resolution=(args.size, args.size),
        device=device,
    )

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    fid_score = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    kid_mean, kid_std = compute_kid(real_features, gen_features)

    print("\n" + "=" * 60)
    print("        MINEART DIFFUSION EVALUATION METRICS REPORT")
    print("=" * 60)
    print(f"  Checkpoint:         {checkpoint_path.name} (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"  Evaluation Samples: {args.num_samples} real vs. {args.num_samples} generated")
    print(f"  Sampling Config:    DDIM {args.steps} steps | CFG: {args.guidance}")
    print(f"  Fréchet Inception Distance (FID): {fid_score:.2f}")
    print(f"  Kernel Inception Distance (KID):  {kid_mean:.4f} ± {kid_std:.4f}")
    print("=" * 60 + "\n")

    results["trained_model"] = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "loss": checkpoint.get("loss", None),
        "fid": fid_score,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }

    # Save sample images
    out_dir = Path(args.save_samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(sample_images):
        img.save(out_dir / f"eval_sample_{idx:02d}.png")
    print(f"[OK] Saved {len(sample_images)} preview samples to: {out_dir}")

    # Save JSON report
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved evaluation metrics to: {out_json}")


if __name__ == "__main__":
    main()
