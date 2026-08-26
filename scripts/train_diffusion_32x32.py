"""
MineArt Diffusion - Phase 2: Structural DDPM for 32x32 Minecraft Artwork

Optimized for strong block geometry, crisp silhouettes, hard edges,
and recognizable scene compositions (terrain, trees, structures, skies).
Features:
  - Multi-Head Spatial Self-Attention for global structural coherence
  - Block-preserving Nearest-Neighbor downsampling (preserves pixel art lines)
  - Improved Cosine Variance Schedule for signal retention
  - Exponential Moving Average (EMA) shadow weights for sharp sampling
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. Dataset Loader with Block-Preserving Sampling
# ==========================================
class MinecraftDataset(Dataset):
    def __init__(self, data_dir: Path, image_size: int = 32, preload_ram: bool = True):
        self.image_paths = sorted(
            [p for p in data_dir.rglob("*.png")] + [p for p in data_dir.rglob("*.jpg")] + [p for p in data_dir.rglob("*.jpeg")]
        )
        self.image_size = image_size
        self.preload_ram = preload_ram
        self.cached_tensors: List[torch.Tensor] = []

        if preload_ram and len(self.image_paths) > 0:
            print(f"Pre-caching {len(self.image_paths)} images into RAM with block-preserving sampling...")
            t0 = time.time()

            def load_single(p: Path) -> Optional[torch.Tensor]:
                try:
                    with Image.open(p) as img:
                        img = img.convert("RGB")
                        # 1. Center-crop to square preserving macro silhouette
                        w, h = img.size
                        min_dim = min(w, h)
                        left = (w - min_dim) // 2
                        top = (h - min_dim) // 2
                        img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
                        
                        # 2. Block-preserving downsampling (NEAREST preserves pixel grid & block contrast)
                        img_32 = img_cropped.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                        
                        arr = np.array(img_32, dtype=np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
                        return torch.from_numpy(arr).permute(2, 0, 1)          # [C, H, W]
                except Exception:
                    return None

            max_threads = min(os.cpu_count() or 8, 16)
            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                results = list(ex.map(load_single, self.image_paths))

            self.cached_tensors = [t for t in results if t is not None]
            elapsed = time.time() - t0
            print(f"Loaded {len(self.cached_tensors)} valid images into RAM in {elapsed:.2f}s!")

    def __len__(self) -> int:
        if self.preload_ram and len(self.cached_tensors) > 0:
            return len(self.cached_tensors)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preload_ram and len(self.cached_tensors) > 0:
            tensor = self.cached_tensors[idx].clone()
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2])  # Horizontal flip
            return tensor

        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
            img_32 = img_cropped.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            if random.random() > 0.5:
                img_32 = img_32.transpose(Image.FLIP_LEFT_RIGHT)
            arr = np.array(img_32, dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1)


# ==========================================
# 2. Sinusoidal Time Positional Embedding
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==========================================
# 3. Spatial Self-Attention for Global Structure
# ==========================================
class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer enabling the model to connect distant
    spatial regions (e.g. skies above, horizon in middle, terrain/structures below).
    """
    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).swapaxes(1, 2)
        x_ln = self.ln(x_flat)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_flat
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(B, C, H, W)


# ==========================================
# 4. ResNet Block with Time Conditioning
# ==========================================
class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.residual(x)


# ==========================================
# 5. Structural UNet for 32x32 Diffusion
# ==========================================
class UNet32(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial projection: 32x32
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        # Downsampling: 32x32 -> 16x16 -> 8x8
        self.down1 = Block(32, 64, time_dim)
        self.pool1 = nn.Conv2d(64, 64, 4, 2, 1)

        self.down2 = Block(64, 128, time_dim)
        self.attn_down = SelfAttention(128, 16)
        self.pool2 = nn.Conv2d(128, 128, 4, 2, 1)

        # Bottleneck: 8x8 with Self-Attention
        self.bot1 = Block(128, 128, time_dim)
        self.bot_attn = SelfAttention(128, 8)
        self.bot2 = Block(128, 128, time_dim)

        # Upsampling: 8x8 -> 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up_block1 = Block(192, 64, time_dim)
        self.attn_up = SelfAttention(64, 16)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up_block2 = Block(96, 32, time_dim)

        # Final projection back to RGB
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timestep)
        x_init = self.init_conv(x)  # [B, 32, 32, 32]

        d1 = self.down1(x_init, t)  # [B, 64, 32, 32]
        p1 = self.pool1(d1)         # [B, 64, 16, 16]

        d2 = self.down2(p1, t)      # [B, 128, 16, 16]
        d2 = self.attn_down(d2)
        p2 = self.pool2(d2)         # [B, 128, 8, 8]

        b1 = self.bot1(p2, t)       # [B, 128, 8, 8]
        b_mid = self.bot_attn(b1)   # Global spatial coherence
        b2 = self.bot2(b_mid, t)    # [B, 128, 8, 8]

        u1 = self.up1(b2)           # [B, 64, 16, 16]
        u1 = self.up_block1(torch.cat([u1, d2], dim=1), t)  # [B, 64, 16, 16]
        u1 = self.attn_up(u1)

        u2 = self.up2(u1)           # [B, 32, 32, 32]
        u2 = self.up_block2(torch.cat([u2, d1], dim=1), t)  # [B, 32, 32, 32]

        return self.final_conv(u2)


# ==========================================
# 6. Gaussian Diffusion with Cosine Schedule
# ==========================================
def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.02) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal) for preserving macro structure
    and edge boundaries deeper into the reverse diffusion trajectory.
    Capped at max_beta to prevent numerical divergence.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, max_beta)


class Diffusion:
    def __init__(self, timesteps: int = 200, schedule: str = "cosine", device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        if schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps, max_beta=0.02).to(device)
        else:
            self.betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32, device=device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: inject noise according to cosine schedule."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        noisy_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Reverse diffusion step: denoise x_t into x_{t-1} with clamped predictions."""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        model_mean = model_mean.clamp(-1.0, 1.0)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n_samples: int = 16, image_size: int = 32) -> torch.Tensor:
        """Generate new images starting from pure Gaussian noise."""
        model.eval()
        x = torch.randn(n_samples, 3, image_size, image_size, device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)

        # Unnormalize from [-1, 1] to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2.0
        return x



# ==========================================
# 7. EMA (Exponential Moving Average)
# ==========================================
class EMA:
    def __init__(self, beta: float = 0.995):
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# ==========================================
# 8. Training and Evaluation Function
# ==========================================
def train_and_generate(
    data_dir: str = "data/raw/screenshots/screenshots",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    timesteps: int = 250,
    output_dir: str = "output_samples",
    image_size: int = 32,
    device: str = "cpu",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_dir = Path(data_dir)
    if not resolved_dir.exists():
        fallback = Path("data/raw")
        if fallback.exists():
            resolved_dir = fallback

    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)

    dataset = MinecraftDataset(resolved_dir, image_size=image_size, preload_ram=True)
    if len(dataset) == 0:
        print(f"Error: No images found in {resolved_dir}.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print("==================================================")
    print("  MineArt Structural DDPM - 32x32 Training Loop  ")
    print("==================================================")
    print(f"Device           : {device}")
    print(f"Dataset Directory : {resolved_dir}")
    print(f"Dataset Size     : {len(dataset)} images")
    print(f"Image Size       : {image_size}x{image_size}")
    print(f"Batch Size       : {batch_size}")
    print(f"Epochs           : {epochs}")
    print(f"Diffusion Steps  : {timesteps} (Cosine Schedule)")
    print("--------------------------------------------------")

    model = UNet32(in_channels=3, out_channels=3, time_dim=64).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMA(beta=0.995)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    diffusion = Diffusion(timesteps=timesteps, schedule="cosine", device=device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (with Self-Attention)")

    loss_history = []
    model_save_path = output_path / "ddpm_minecraft_32x32.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0_epoch = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs:02d}")
        for batch in pbar:
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()

            # Forward diffusion with cosine noise
            x_noisy, noise = diffusion.q_sample(batch, t)

            # Predict added noise
            predicted_noise = model(x_noisy, t)
            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update_model_average(ema_model, model)

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        epoch_time = time.time() - t0_epoch
        print(f"Epoch {epoch:02d}/{epochs:02d} - Average MSE Loss: {avg_loss:.5f} ({epoch_time:.1f}s)")

        # Save EMA weights for crisp inference
        torch.save(ema_model.state_dict(), model_save_path)

        # Intermediate sampling
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Generating sample 32x32 Minecraft images for Epoch {epoch}...")
            samples = diffusion.sample(ema_model, n_samples=16, image_size=image_size)
            save_sample_grid(samples, output_path / f"generated_epoch_{epoch:02d}.png", title=f"Generated 32x32 Minecraft Art (Epoch {epoch})")

    print(f"\n-> Saved trained EMA model weights to: {model_save_path}")

    # Plot training loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), loss_history, marker="o", color="#2196F3", label="MSE Loss")
    plt.title("DDPM Training Loss Curve (Structural Minecraft Model)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    loss_curve_path = output_path / "training_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"-> Saved loss curve to: {loss_curve_path}")

    # Generate final high-quality grid
    print("\nGenerating final 4x4 grid of 32x32 Minecraft artwork...")
    final_samples = diffusion.sample(ema_model, n_samples=16, image_size=image_size)
    save_sample_grid(final_samples, output_path / "final_generated_minecraft_samples.png", title="MineArt Diffusion: Structural 32x32 Minecraft Artwork")
    print(f"-> Saved final generated artwork grid to: {output_path / 'final_generated_minecraft_samples.png'}")


def save_sample_grid(samples: torch.Tensor, output_path: Path, title: str = "Generated Samples"):
    """Save a 4x4 grid of generated 32x32 image tensors."""
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(16):
        axes[i].imshow(samples[i])
        axes[i].axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="MineArt Diffusion - Structural DDPM Training on 32x32 Minecraft Screenshots")
    parser.add_argument("--data-dir", type=str, default="data/raw/screenshots/screenshots", help="Path to raw screenshots directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--timesteps", type=int, default=250, help="Diffusion timesteps (default: 250)")
    parser.add_argument("--output-dir", type=str, default="output_samples", help="Output directory for generated art")
    parser.add_argument("--size", type=int, default=32, help="Image resolution width/height (default: 32)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_generate(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        image_size=args.size,
        device=device,
    )
