# MineArt Diffusion — Project Context & Architecture

## 1. Project Overview

**Project Name:** MineArt Diffusion

MineArt Diffusion is a machine-learning project focused on training a **custom diffusion model from scratch** to generate authentic Minecraft-style artwork.

### Core Philosophy
- **No External Foundation Models**: We are intentionally NOT using pretrained SDXL, FLUX, or external commercial image APIs.
- **Custom ML from Scratch**: The primary academic and engineering objective is building, training, and optimizing a custom Denoising Diffusion Probabilistic Model (DDPM) directly on a target dataset of ~50,000 Minecraft images.
- **Minimal, Maintainable Architecture**: Keep the project clean, focused, and free of premature abstractions.

---

## 2. End-to-End Pipeline

```text
Minecraft Raw Images (~50,000 target)
                 ↓
      Dataset Preprocessing
(Cleaning, validation, normalization, train/val/test splits)
                 ↓
     Custom Diffusion Model (DDPM)
(U-Net with Sinusoidal Time Positional Embeddings)
                 ↓
     Minecraft Artwork Generation
```

---

## 3. Phased Roadmap

### Phase 1: Dataset Collection & Engineering (CURRENT)
- Collect raw in-game screenshots and video gameplay frames into `data/raw/` (target: ~50,000 images).
- Automated validation, corrupt image detection, HUD/low-entropy filtering, and deduplication.
- Generate clean, standardized dataset splits in `data/processed/`.

### Phase 2: Custom Diffusion Model Development
- Develop and train custom PyTorch DDPM architecture on processed Minecraft images.
- Optimize beta schedules (linear / cosine), timestep embeddings, and U-Net depth.
- Evaluate loss curves, FID scores, and sample quality.

### Later Future Phases (Deferred)
- Environment recognition & biome classification
- Mob and object detection
- Semantic segmentation
- Canonical Minecraft painting export & resource pack formatting
- Web application / UI interface
