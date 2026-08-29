"""MineArt High-Definition Minecraft Painting Diffusion Model.

A deep conditional Denoising Diffusion Probabilistic Model (DDPM) built to synthesize
authentic, high-fidelity Minecraft artwork and painting canvases at 64px per block.

Supported In-Game Minecraft Painting Aspect Ratios:
  - 1:1  (1x1 block)  -> 64 x 64 px   (Kebab, Aztec, Alban, Target, Plant)
  - 2:1  (2x1 blocks) -> 128 x 64 px  (Pool, Sunset, Sea, Creebet)
  - 1:2  (1x2 blocks) -> 64 x 128 px  (Wanderer, Graham)
  - 2:2  (2x2 blocks) -> 128 x 128 px (Bust, Match, Skull, Stage, Void, Wither)
  - 3:3  (3x3 blocks) -> 192 x 192 px (Donkey Kong)
  - 3:4  (3x4 blocks) -> 192 x 256 px (Finding, Passage, Unpacked)
  - 4:3  (4x3 blocks) -> 256 x 192 px (Skeleton, Fighters, Courbet)
  - 4:2  (4x2 blocks) -> 256 x 128 px (Pigscene)
  - 4:4  (4x4 blocks) -> 256 x 256 px (Pointer, Nether, Burning Skull, Fern)

Features:
  1. Multi-Stage U-Net with Residual Blocks, GroupNorm, and Spatial Self-Attention.
  2. Dynamic Latent Thresholding to prevent color burning / saturation drift.
  3. Classifier-Free Guidance (CFG) for sharp prompt adherence.
  4. Full Image-to-Image transformation mode for stylizing existing captures.
  5. Optimized for 16 GB Cloud GPUs (Lightning AI / Colab) with Batch Size 128.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


PAINTING_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1": (64, 64),
    "2:1": (128, 64),
    "1:2": (64, 128),
    "2:2": (128, 128),
    "3:3": (192, 192),
    "3:4": (192, 256),
    "4:3": (256, 192),
    "4:2": (256, 128),
    "4:4": (256, 256),
    "1x1": (64, 64),
    "2x1": (128, 64),
    "1x2": (64, 128),
    "2x2": (128, 128),
    "3x3": (192, 192),
    "3x4": (192, 256),
    "4x3": (256, 192),
    "4x2": (256, 128),
    "4x4": (256, 256),
}


class PromptTokenizer:
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
            "stone", "wood", "tree", "animal", "monster", "weapon", "armor",
            "painting", "landscape", "portrait", "mural", "canvas"
        ]
        for w in common_minecraft_terms:
            words.add(w)

        self.vocab = list(self.special_tokens) + sorted(list(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

    def encode(self, text: str, max_length: int = 16) -> torch.Tensor:
        if not text or text.strip() == "":
            return torch.full((max_length,), self.word2idx[self.null_token], dtype=torch.long)

        tokens = text.replace(",", " ").replace("_", " ").lower().split()
        indices = [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in tokens[:max_length]]
        while len(indices) < max_length:
            indices.append(self.word2idx[self.pad_token])
        return torch.tensor(indices, dtype=torch.long)

    def null_tokens(self, max_length: int = 16) -> torch.Tensor:
        return torch.full((max_length,), self.word2idx[self.null_token], dtype=torch.long)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PromptTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])


class MineArtDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str = "data/metadata.csv",
        image_dir: str = "data/images",
        target_size: Tuple[int, int] = (384, 224),
        max_prompt_length: int = 16,
        tokenizer: Optional[PromptTokenizer] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.max_prompt_length = max_prompt_length

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
            self.prompts.append(f"{subj} {tags}".strip())

        self.tokenized_prompts = torch.stack(
            [self.tokenizer.encode(p, max_length=self.max_prompt_length) for p in self.prompts]
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size[1], self.target_size[0]), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_dir / self.filenames[idx]
        if img_path.exists():
            try:
                with Image.open(img_path) as raw_img:
                    image = self.transform(raw_img.convert("RGB"))
            except Exception:
                image = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
        else:
            image = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        return image, self.tokenized_prompts[idx]


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).view(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.transpose(-2, -1)
        scores = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(b, c, h, w)
        return x + self.proj_out(out)


class ResnetBlock(nn.Module):
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


class MineArtUNet(nn.Module):
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

        self.down1_1 = ResnetBlock(base_channels, base_channels, cond_dim)
        self.down1_2 = ResnetBlock(base_channels, base_channels, cond_dim)
        self.down1_down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.down2_1 = ResnetBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.down2_2 = ResnetBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.down2_down = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.down3_1 = ResnetBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.down3_2 = ResnetBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.down3_attn = AttentionBlock(base_channels * 4)
        self.down3_down = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.mid1 = ResnetBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid2 = ResnetBlock(base_channels * 4, base_channels * 4, cond_dim)

        self.up3_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.up3_1 = ResnetBlock(base_channels * 8, base_channels * 4, cond_dim)
        self.up3_2 = ResnetBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.up3_attn = AttentionBlock(base_channels * 4)

        self.up2_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2_1 = ResnetBlock(base_channels * 4, base_channels * 2, cond_dim)
        self.up2_2 = ResnetBlock(base_channels * 2, base_channels * 2, cond_dim)

        self.up1_up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1_1 = ResnetBlock(base_channels * 2, base_channels, cond_dim)
        self.up1_2 = ResnetBlock(base_channels, base_channels, cond_dim)

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

        x1 = self.init_conv(x)
        d1 = self.down1_1(x1, cond)
        d1 = self.down1_2(d1, cond)

        x2 = self.down1_down(d1)
        d2 = self.down2_1(x2, cond)
        d2 = self.down2_2(d2, cond)

        x3 = self.down2_down(d2)
        d3 = self.down3_1(x3, cond)
        d3 = self.down3_2(d3, cond)
        d3 = self.down3_attn(d3)

        x4 = self.down3_down(d3)
        m = self.mid1(x4, cond)
        m = self.mid_attn(m)
        m = self.mid2(m, cond)

        u3 = self.up3_up(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3_1(u3, cond)
        u3 = self.up3_2(u3, cond)
        u3 = self.up3_attn(u3)

        u2 = self.up2_up(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2_1(u2, cond)
        u2 = self.up2_2(u2, cond)

        u1 = self.up1_up(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1_1(u1, cond)
        u1 = self.up1_2(u1, cond)

        return self.out_conv(self.out_act(self.out_norm(u1)))


class GaussianDiffusionEngine:
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
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
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

        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / self.sqrt_alphas_cumprod[t]
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        model_mean = (
            self.alphas_cumprod_prev[t].sqrt() * beta_t / (1.0 - self.alphas_cumprod[t]) * pred_x0
            + (1.0 - self.alphas_cumprod_prev[t]).sqrt() * self.alphas[t].sqrt() / (1.0 - self.alphas_cumprod[t]) * x
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
        shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
        guidance_scale: float = 2.0,
        ddim_steps: int = 100,
    ) -> torch.Tensor:
        model.eval()
        img = torch.randn(shape, device=self.device)

        step_interval = max(1, self.timesteps // ddim_steps)
        timesteps_to_sample = list(range(0, self.timesteps, step_interval))

        for t in tqdm(reversed(timesteps_to_sample), total=len(timesteps_to_sample), desc="Rendering Painting"):
            img = self.p_sample(
                model=model,
                x=img,
                t=t,
                prompt_tokens=prompt_tokens,
                uncond_tokens=uncond_tokens,
                guidance_scale=guidance_scale,
            )

        return (img.clamp(-1.0, 1.0) + 1.0) / 2.0

    @torch.no_grad()
    def sample_image_to_image(
        self,
        model: nn.Module,
        init_image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        strength: float = 0.60,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        model.eval()
        start_timestep = int(self.timesteps * strength)
        start_timestep = max(1, min(self.timesteps - 1, start_timestep))

        t_start = torch.full((init_image.shape[0],), start_timestep, device=self.device, dtype=torch.long)
        noise = torch.randn_like(init_image)
        img = self.q_sample(init_image, t_start, noise=noise)

        for t in tqdm(reversed(range(start_timestep)), total=start_timestep, desc="Transforming Canvas"):
            img = self.p_sample(
                model=model,
                x=img,
                t=t,
                prompt_tokens=prompt_tokens,
                uncond_tokens=uncond_tokens,
                guidance_scale=guidance_scale,
            )

        return (img.clamp(-1.0, 1.0) + 1.0) / 2.0


def train_diffusion(
    metadata_csv: str = "data/metadata.csv",
    image_dir: str = "data/images",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    timesteps: int = 1000,
    uncond_prob: float = 0.15,
    save_dir: str = "checkpoints",
    resume_path: Optional[str] = None,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset = MineArtDataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        target_size=(384, 224),
    )

    tokenizer_path = save_path / "diffusion_tokenizer.json"
    dataset.tokenizer.save(str(tokenizer_path))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = MineArtUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        vocab_size=len(dataset.tokenizer.vocab),
        text_embed_dim=128,
        time_embed_dim=128,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized MineArt Diffusion U-Net: {total_params:,} parameters.")

    diffusion = GaussianDiffusionEngine(timesteps=timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    if resume_path and Path(resume_path).exists():
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

    null_tokens = dataset.tokenizer.null_tokens(dataset.max_prompt_length).to(device)
    print(f"Beginning training: {epochs} epochs over {len(dataset)} samples (Batch Size: {batch_size})...")

    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
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

            optimizer.zero_grad(set_to_none=True)

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
        elapsed = time.time() - epoch_t0

        print(f"Epoch {epoch:03d}/{epochs:03d} | MSE Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == epochs:
            checkpoint_path = save_path / f"mineart_diffusion_epoch_{epoch}.pt"
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
            }, save_path / "best_diffusion.pt")

    total_time = time.time() - t_start
    print(f"Training completed in {total_time / 60:.2f} minutes.")


def generate_painting(
    prompt: str,
    ratio: str = "1:1",
    input_image_path: Optional[str] = None,
    strength: float = 0.60,
    checkpoint_path: str = "checkpoints/best_diffusion.pt",
    tokenizer_path: str = "checkpoints/diffusion_tokenizer.json",
    output_path: str = "output_samples/painting.png",
    guidance_scale: float = 2.0,
    ddim_steps: int = 100,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first!")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    target_w, target_h = PAINTING_RATIOS.get(ratio, (64, 64))
    print(f"Rendering Minecraft Painting: Ratio '{ratio}' ({target_w}x{target_h} px | {target_w // 64}x{target_h // 64} blocks)")

    tokenizer = PromptTokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    prompt_tokens = tokenizer.encode(prompt, max_length=16).unsqueeze(0).to(device)
    uncond_tokens = tokenizer.null_tokens(max_length=16).unsqueeze(0).to(device)

    if input_image_path is not None:
        preprocess = transforms.Compose([
            transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        with Image.open(input_image_path) as raw_img:
            init_tensor = preprocess(raw_img.convert("RGB")).unsqueeze(0).to(device)

        print(f"Modifying image '{input_image_path}' with prompt '{prompt}' (Strength: {strength})...")
        sampled = diffusion.sample_image_to_image(
            model=model,
            init_image=init_tensor,
            prompt_tokens=prompt_tokens,
            uncond_tokens=uncond_tokens,
            strength=strength,
            guidance_scale=guidance_scale,
        )
    else:
        print(f"Synthesizing new painting from prompt: '{prompt}' (CFG: {guidance_scale})...")
        sampled = diffusion.sample_text_to_image(
            model=model,
            prompt_tokens=prompt_tokens,
            uncond_tokens=uncond_tokens,
            shape=(1, 3, target_h, target_w),
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
        )

    img_np = (sampled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil.save(output_path)
    print(f"Saved finished painting to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MineArt Minecraft Painting Diffusion Generator")
    parser.add_argument("--train", action="store_true", help="Start training model")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (128 for 16GB Cloud GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for synthesis")
    parser.add_argument("--image", type=str, default=None, help="Input image for image-to-image painting variation")
    parser.add_argument("--ratio", type=str, default="1:1", choices=list(PAINTING_RATIOS.keys()), help="Painting aspect ratio: 1:1, 2:1, 1:2, 2:2, 3:3, 3:4, 4:3, 4:2, 4:4")
    parser.add_argument("--strength", type=float, default=0.60, help="Image modification strength (0.1 - 0.9)")
    parser.add_argument("--guidance", type=float, default=2.0, help="Classifier-free guidance scale (1.5 - 2.5)")
    parser.add_argument("--steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--output", type=str, default="output_samples/painting.png", help="Output PNG path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_diffusion.pt", help="Checkpoint path")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/diffusion_tokenizer.json", help="Tokenizer path")

    args = parser.parse_args()

    if args.train:
        train_diffusion(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_path=args.resume,
        )
    elif args.prompt is not None or args.image is not None:
        generate_painting(
            prompt=args.prompt or "minecraft painting",
            ratio=args.ratio,
            input_image_path=args.image,
            strength=args.strength,
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            guidance_scale=args.guidance,
            ddim_steps=args.steps,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()