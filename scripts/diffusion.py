"""MineArt Diffusion 16x16 Generative Model.

This module implements a complete conditional Denoising Diffusion Probabilistic
Model (DDPM) designed to synthesize novel, authentic 16x16 Minecraft-style pixel
artwork. It supports both text-prompt guided generation (Text-to-Image) and
image-conditioned variation (Image-to-Image) using Classifier-Free Guidance.

Key Capabilities:
1. Novel Image Synthesis: Generates distinct Minecraft imagery not present in
   the training dataset by sampling from learned continuous reverse diffusion trajectories.
2. Dual Conditioning:
   - Text Conditioning: Encodes textual prompts, subjects, and biome/entity tags
     into latent conditioning vectors.
   - Image Conditioning: Performs image-to-image diffusion by partially noising
     a user-provided source image to an intermediate timestep and denoising it
     according to the target prompt.
3. Classifier-Free Guidance (CFG): Supports unconditional dropout during training
   to enable adjustable prompt adherence strength during generation.
4. Memory-Optimized In-RAM Dataset: Caches 16x16 downscaled representations of
   the entire dataset in memory to eliminate disk I/O bottlenecks and achieve
   fast per-epoch training times.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# pyrefly: ignore
from torchvision import transforms
from tqdm import tqdm


class PromptTokenizer:
    """Builds and tokenizes multi-word Minecraft prompts and metadata tags.

    Maps subject names, biome descriptors, entity labels, and common visual
    tokens into indices for learned embedding representations.
    """

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.null_token = "<null>"
        self.special_tokens = [self.pad_token, self.unk_token, self.null_token]

        if vocab is not None:
            self.vocab = list(self.special_tokens) + [w for w in vocab if w not in self.special_tokens]
        else:
            self.vocab = list(self.special_tokens)

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """Constructs vocabulary from subjects and comma-separated tags."""
        words = set()
        for subj in df["subject"].dropna().unique():
            for piece in str(subj).replace("_", " ").split():
                clean = piece.strip().lower()
                if clean:
                    words.add(clean)

        for tag_str in df["tags"].dropna().unique():
            for t in str(tag_str).split(","):
                for piece in t.replace("_", " ").split():
                    clean = piece.strip().lower()
                    if clean:
                        words.add(clean)

        common_minecraft_terms = [
            "minecraft", "art", "pixel", "block", "scene", "mob", "terrain",
            "sunrise", "sunset", "night", "day", "water", "lava", "sky", "grass",
            "stone", "wood", "tree", "animal", "monster", "weapon", "armor"
        ]
        for w in common_minecraft_terms:
            words.add(w)

        self.vocab = list(self.special_tokens) + sorted(list(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

    def encode(self, text: str, max_length: int = 16) -> torch.Tensor:
        """Converts a text string into a fixed-length tensor of token indices."""
        if not text or text.strip() == "":
            return torch.full((max_length,), self.word2idx[self.null_token], dtype=torch.long)

        tokens = text.replace(",", " ").replace("_", " ").lower().split()
        indices = [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in tokens[:max_length]]
        while len(indices) < max_length:
            indices.append(self.word2idx[self.pad_token])
        return torch.tensor(indices, dtype=torch.long)

    def null_tokens(self, max_length: int = 16) -> torch.Tensor:
        """Returns the unconditional null token sequence."""
        return torch.full((max_length,), self.word2idx[self.null_token], dtype=torch.long)

    def save(self, filepath: str) -> None:
        """Serializes the tokenizer vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PromptTokenizer":
        """Loads a tokenizer instance from a saved vocabulary JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])


class MineArt16x16DiffusionDataset(Dataset):
    """PyTorch Dataset with RAM caching for 16x16 diffusion training.

    Resizes images to 16x16 normalized in [-1, 1] and tokenizes combined
    subject and tag descriptions for conditioned text-to-image training.
    """

    def __init__(
        self,
        metadata_csv: str = "data/metadata.csv",
        image_dir: str = "data/images",
        target_size: int = 16,
        max_prompt_length: int = 16,
        tokenizer: Optional[PromptTokenizer] = None,
        cache_in_ram: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.max_prompt_length = max_prompt_length
        self.cache_in_ram = cache_in_ram

        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

        self.df = pd.read_csv(metadata_csv)

        if tokenizer is None:
            self.tokenizer = PromptTokenizer()
            self.tokenizer.build_from_dataframe(self.df)
        else:
            self.tokenizer = tokenizer

        self.filenames = self.df["filename"].tolist()
        self.prompts = []
        for _, row in self.df.iterrows():
            subj = str(row["subject"]).replace("_", " ") if pd.notna(row["subject"]) else ""
            tags = str(row["tags"]) if pd.notna(row["tags"]) else ""
            combined_prompt = f"{subj} {tags}".strip()
            self.prompts.append(combined_prompt)

        self.tokenized_prompts = torch.stack(
            [self.tokenizer.encode(p, max_length=self.max_prompt_length) for p in self.prompts]
        )

        self.preprocess = transforms.Compose([
            transforms.CenterCrop((1080, 1080)),
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.cached_images: Optional[torch.Tensor] = None
        if self.cache_in_ram:
            self._preload_dataset()

    def _preload_dataset(self) -> None:
        """Pre-processes and loads all 16x16 frames into a continuous RAM tensor."""
        t0 = time.time()
        tensors = []
        blank_tensor = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)

        for fname in tqdm(self.filenames, desc="Preloading 16x16 Frames"):
            img_path = self.image_dir / fname
            if not img_path.exists():
                tensors.append(blank_tensor)
                continue
            try:
                with Image.open(img_path) as img:
                    tensor = self.preprocess(img.convert("RGB"))
                    tensors.append(tensor)
            except Exception:
                tensors.append(blank_tensor)

        self.cached_images = torch.stack(tensors)
        ram_mb = (self.cached_images.element_size() * self.cached_images.nelement()) / (1024 * 1024)
        elapsed = time.time() - t0
        print(f"Preloaded {len(self.filenames)} images in {elapsed:.2f}s ({ram_mb:.2f} MB in RAM).")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cached_images is not None:
            image = self.cached_images[idx]
        else:
            img_path = self.image_dir / self.filenames[idx]
            if img_path.exists():
                with Image.open(img_path) as img:
                    image = self.preprocess(img.convert("RGB"))
            else:
                image = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)

        prompt_tokens = self.tokenized_prompts[idx]
        return image, prompt_tokens


class SinusoidalPositionEmbeddings(nn.Module):
    """Computes sinusoidal timestep embeddings for diffusion models."""

    def __init__(self, dim: int) -> None:
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


class AttentionBlock(nn.Module):
    """Self-attention block for spatial feature correlation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        q = self.q(norm_x).view(b, c, h * w).transpose(1, 2)
        k = self.k(norm_x).view(b, c, h * w)
        v = self.v(norm_x).view(b, c, h * w).transpose(1, 2)

        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class ConditionedResBlock(nn.Module):
    """Residual convolutional block with additive timestep and text embeddings."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        cond_emb = self.cond_proj(cond)[:, :, None, None]
        h = h + cond_emb
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class MineArt16x16UNet(nn.Module):
    """Conditional U-Net architecture for 16x16 Minecraft diffusion generation.

    Receives a 3-channel noisy 16x16 image tensor alongside sinusoidal
    diffusion timesteps and pooled prompt conditioning embeddings, predicting
    the noise component epsilon.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        vocab_size: int = 256,
        text_embed_dim: int = 128,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(text_embed_dim * 2, text_embed_dim),
        )

        cond_dim = time_embed_dim + text_embed_dim

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1_block1 = ConditionedResBlock(base_channels, base_channels, cond_dim)
        self.down1_block2 = ConditionedResBlock(base_channels, base_channels, cond_dim)
        self.down1_down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.down2_block1 = ConditionedResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.down2_block2 = ConditionedResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.down2_attn = AttentionBlock(base_channels * 2)
        self.down2_down = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.mid_block1 = ConditionedResBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ConditionedResBlock(base_channels * 4, base_channels * 4, cond_dim)

        self.up2_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2_block1 = ConditionedResBlock(base_channels * 4, base_channels * 2, cond_dim)
        self.up2_block2 = ConditionedResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.up2_attn = AttentionBlock(base_channels * 2)

        self.up1_up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1_block1 = ConditionedResBlock(base_channels * 2, base_channels, cond_dim)
        self.up1_block2 = ConditionedResBlock(base_channels, base_channels, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        text_emb = self.text_embedding(prompt_tokens).mean(dim=1)
        text_emb = self.text_proj(text_emb)
        cond = torch.cat([time_emb, text_emb], dim=-1)

        x = self.init_conv(x)
        d1 = self.down1_block1(x, cond)
        d1 = self.down1_block2(d1, cond)

        x2 = self.down1_down(d1)
        d2 = self.down2_block1(x2, cond)
        d2 = self.down2_block2(d2, cond)
        d2 = self.down2_attn(d2)

        x3 = self.down2_down(d2)
        m = self.mid_block1(x3, cond)
        m = self.mid_attn(m)
        m = self.mid_block2(m, cond)

        u2 = self.up2_up(m)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2_block1(u2, cond)
        u2 = self.up2_block2(u2, cond)
        u2 = self.up2_attn(u2)

        u1 = self.up1_up(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1_block1(u1, cond)
        u1 = self.up1_block2(u1, cond)

        out = self.out_conv(self.out_act(self.out_norm(u1)))
        return out


class GaussianDiffusionEngine:
    """Manages forward noise addition and reverse sampling trajectories.

    Implements a cosine noise variance schedule, Classifier-Free Guidance,
    and partial-noise image-to-image synthesis.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adds noise to the original images according to timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        prompt_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """Performs a single reverse diffusion step with Classifier-Free Guidance."""
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        if guidance_scale > 1.0:
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            prompt_double = torch.cat([prompt_tokens, uncond_tokens], dim=0)

            pred_noise_double = model(x_double, t_double, prompt_double)
            cond_noise, uncond_noise = pred_noise_double.chunk(2, dim=0)
            pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
        else:
            pred_noise = model(x, t_tensor, prompt_tokens)

        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t])

        model_mean = sqrt_recip_alpha_t * (
            x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(self.posterior_variance[t])
            return model_mean + sigma_t * noise
        return model_mean

    @torch.no_grad()
    def sample_text_to_image(
        self,
        model: nn.Module,
        prompt_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        shape: Tuple[int, int, int, int] = (1, 3, 16, 16),
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """Generates novel 16x16 images starting from pure Gaussian noise."""
        model.eval()
        img = torch.randn(shape, device=self.device)

        for t in reversed(range(self.timesteps)):
            img = self.p_sample(
                model=model,
                x=img,
                t=t,
                prompt_tokens=prompt_tokens,
                uncond_tokens=uncond_tokens,
                guidance_scale=guidance_scale,
            )

        img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
        return img

    @torch.no_grad()
    def sample_image_to_image(
        self,
        model: nn.Module,
        init_image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        strength: float = 0.65,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """Generates a novel variation of an input image guided by a prompt."""
        model.eval()
        start_timestep = int(self.timesteps * strength)
        start_timestep = max(1, min(self.timesteps - 1, start_timestep))

        t_start = torch.full((init_image.shape[0],), start_timestep, device=self.device, dtype=torch.long)
        noise = torch.randn_like(init_image)
        img = self.q_sample(init_image, t_start, noise=noise)

        for t in reversed(range(start_timestep)):
            img = self.p_sample(
                model=model,
                x=img,
                t=t,
                prompt_tokens=prompt_tokens,
                uncond_tokens=uncond_tokens,
                guidance_scale=guidance_scale,
            )

        img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
        return img


def train_diffusion(
    metadata_csv: str = "data/metadata.csv",
    image_dir: str = "data/images",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 2e-4,
    timesteps: int = 1000,
    uncond_prob: float = 0.15,
    save_dir: str = "checkpoints",
) -> None:
    """Trains the conditional 16x16 diffusion model using in-memory cached frames."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = MineArt16x16DiffusionDataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        target_size=16,
        cache_in_ram=True,
    )

    tokenizer_path = save_path / "diffusion_16x16_tokenizer.json"
    dataset.tokenizer.save(str(tokenizer_path))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = MineArt16x16UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        vocab_size=len(dataset.tokenizer.vocab),
        text_embed_dim=128,
        time_embed_dim=128,
    ).to(device)

    diffusion = GaussianDiffusionEngine(timesteps=timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    null_tokens = dataset.tokenizer.null_tokens(dataset.max_prompt_length).to(device)
    loss_history = []

    print(f"Initialized MineArt Diffusion U-Net ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"Beginning training for {epochs} epochs over {len(dataset)} samples...")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_t0 = time.time()

        for images, prompt_tokens in loader:
            images = images.to(device, non_blocking=True)
            prompt_tokens = prompt_tokens.to(device, non_blocking=True)

            if uncond_prob > 0.0:
                mask = torch.rand(prompt_tokens.shape[0], device=device) < uncond_prob
                prompt_tokens[mask] = null_tokens

            b = images.shape[0]
            t = torch.randint(0, timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, t, noise=noise)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    pred_noise = model(noisy_images, t, prompt_tokens)
                    loss = F.mse_loss(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_noise = model(noisy_images, t, prompt_tokens)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * b

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        elapsed = time.time() - epoch_t0

        print(f"Epoch {epoch:03d}/{epochs:03d} | MSE Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = save_path / f"mineart_diffusion_16x16_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "vocab_size": len(dataset.tokenizer.vocab),
            }, checkpoint_path)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "vocab_size": len(dataset.tokenizer.vocab),
            }, save_path / "best_diffusion_16x16.pt")

    total_time = time.time() - t_start
    print(f"Diffusion model training complete in {total_time / 60:.2f} minutes.")


def generate_text_to_image(
    prompt: str,
    checkpoint_path: str = "checkpoints/best_diffusion_16x16.pt",
    tokenizer_path: str = "checkpoints/diffusion_16x16_tokenizer.json",
    output_path: str = "output_samples/generated_16x16.png",
    guidance_scale: float = 3.5,
    upscale_factor: int = 16,
) -> str:
    """Synthesizes a novel 16x16 Minecraft image from a text prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = PromptTokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MineArt16x16UNet(
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

    prompt_tokens = tokenizer.encode(prompt, max_length=16).unsqueeze(0).to(device)
    uncond_tokens = tokenizer.null_tokens(max_length=16).unsqueeze(0).to(device)

    print(f"Synthesizing image for prompt: '{prompt}' (CFG Scale: {guidance_scale})...")
    sampled = diffusion.sample_text_to_image(
        model=model,
        prompt_tokens=prompt_tokens,
        uncond_tokens=uncond_tokens,
        shape=(1, 3, 16, 16),
        guidance_scale=guidance_scale,
    )

    img_np = (sampled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if upscale_factor > 1:
        upscaled = img_pil.resize(
            (16 * upscale_factor, 16 * upscale_factor),
            resample=Image.Resampling.NEAREST,
        )
        upscaled.save(output_path)
    else:
        img_pil.save(output_path)

    print(f"Generated image saved to: {output_path}")
    return output_path


def generate_image_to_image(
    input_image_path: str,
    prompt: str,
    strength: float = 0.65,
    checkpoint_path: str = "checkpoints/best_diffusion_16x16.pt",
    tokenizer_path: str = "checkpoints/diffusion_16x16_tokenizer.json",
    output_path: str = "output_samples/img2img_16x16.png",
    guidance_scale: float = 3.5,
    upscale_factor: int = 16,
) -> str:
    """Generates a novel variation of an input image guided by a text prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = PromptTokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MineArt16x16UNet(
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

    preprocess = transforms.Compose([
        transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    with Image.open(input_image_path) as raw_img:
        init_tensor = preprocess(raw_img.convert("RGB")).unsqueeze(0).to(device)

    prompt_tokens = tokenizer.encode(prompt, max_length=16).unsqueeze(0).to(device)
    uncond_tokens = tokenizer.null_tokens(max_length=16).unsqueeze(0).to(device)

    print(f"Transforming source image '{input_image_path}' with prompt '{prompt}' (Strength: {strength})...")
    sampled = diffusion.sample_image_to_image(
        model=model,
        init_image=init_tensor,
        prompt_tokens=prompt_tokens,
        uncond_tokens=uncond_tokens,
        strength=strength,
        guidance_scale=guidance_scale,
    )

    img_np = (sampled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if upscale_factor > 1:
        upscaled = img_pil.resize(
            (16 * upscale_factor, 16 * upscale_factor),
            resample=Image.Resampling.NEAREST,
        )
        upscaled.save(output_path)
    else:
        img_pil.save(output_path)

    print(f"Image-to-image output saved to: {output_path}")
    return output_path


def main() -> None:
    """CLI entry point for training and sampling 16x16 MineArt diffusion."""
    parser = argparse.ArgumentParser(description="MineArt 16x16 Diffusion Generator")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epoch count")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for image synthesis")
    parser.add_argument("--image", type=str, default=None, help="Source image for image-to-image variation")
    parser.add_argument("--strength", type=float, default=0.65, help="Image-to-image noise strength [0.1 - 0.9]")
    parser.add_argument("--guidance", type=float, default=3.5, help="Classifier-free guidance scale")
    parser.add_argument("--output", type=str, default="output_samples/output_16x16.png", help="Output file path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_diffusion_16x16.pt", help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/diffusion_16x16_tokenizer.json", help="Tokenizer path")
    parser.add_argument("--upscale", type=int, default=16, help="Pixel art preview upscale factor")

    args = parser.parse_args()

    if args.train:
        train_diffusion(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.image is not None:
        generate_image_to_image(
            input_image_path=args.image,
            prompt=args.prompt or "minecraft scene",
            strength=args.strength,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            guidance_scale=args.guidance,
            upscale_factor=args.upscale,
        )
    elif args.prompt is not None:
        generate_text_to_image(
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            guidance_scale=args.guidance,
            upscale_factor=args.upscale,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
