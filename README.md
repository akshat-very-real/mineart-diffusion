# MineArt Diffusion

MineArt Diffusion is a machine-learning project for training a custom Denoising Diffusion Probabilistic Model (DDPM) from scratch on authentic Minecraft imagery.

---

## 🎯 Project Goal & Architecture

Train a lightweight, custom diffusion model from scratch on a target dataset of **~50,000 Minecraft images** collected from in-game exploration, gameplay recordings, and community sources.

```text
Minecraft Raw Screenshots (~50k target)
                 ↓
      Dataset Preprocessing
(Validation, deduplication, crop/scale, split)
                 ↓
      Custom Diffusion Model
      (DDPM + UNet Architecture)
                 ↓
     Minecraft Artwork Generation
```

> **Note:** MineArt uses **no external pretrained diffusion models** (no SDXL, FLUX, or commercial image APIs). The primary ML contribution is training our own model directly on the Minecraft domain.

---

## 📁 Repository Structure

```text
mineart-diffusion/
│
├── data/
│   ├── raw/                  # Original, unedited Minecraft screenshots
│   └── processed/            # Normalized, cleaned dataset splits (train / val / test)
│
├── scripts/
│   ├── watch_ss.py           # Real-time screenshot watcher from .minecraft
│   ├── extract_frames.py     # Video gameplay frame extractor
│   ├── prepare_dataset.py    # Automated dataset validation, cleaning & split pipeline
│   └── train_diffusion_32x32.py # Custom from-scratch PyTorch DDPM training script
│
├── notebooks/
│   └── dataset_analysis.ipynb # Interactive exploration and dataset analytics
│
├── requirements.txt          # Minimal ML dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Dataset Collection
- **Live Screenshot Watcher**:
  ```bash
  python scripts/watch_ss.py
  ```
- **Video Gameplay Extraction**:
  ```bash
  python scripts/extract_frames.py --video path/to/gameplay.mp4 --interval 2.0
  ```

### 3. Dataset Preprocessing & Cleaning
Run the automated cleaning and split pipeline:
```bash
python scripts/prepare_dataset.py --input-dir data/raw --output-dir data/processed --target-size 32
```

### 4. Custom Diffusion Model Training
Train the custom DDPM model on your processed Minecraft dataset:
```bash
python scripts/train_diffusion_32x32.py --data-dir data/processed/train --epochs 50 --batch-size 32
```
