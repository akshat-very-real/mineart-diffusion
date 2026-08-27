# MineArt Diffusion

MineArt Diffusion is a machine-learning project for training a custom Denoising Diffusion Probabilistic Model (DDPM) from scratch on authentic Minecraft imagery.

---

## Project Goal & Architecture

Train a lightweight, custom diffusion model from scratch on a target dataset of **~50,000 Minecraft images** collected from in-game exploration, high-definition gameplay recordings, and authentic Minecraft world captures.

```text
Minecraft Raw Captures (Target: ~50,000 | Current: 22,600)
                         ↓
               Dataset Preprocessing
   (Validation, deduplication, crop/scale, split)
                         ↓
               Custom Diffusion Model
             (DDPM + UNet Architecture)
                         ↓
            Minecraft Artwork Generation
```

> **Core Philosophy:** MineArt uses **no external pretrained diffusion models** (no SDXL, FLUX, or commercial image APIs). The primary ML contribution is training our own custom PyTorch diffusion architecture directly on the Minecraft domain.

---

## Current Dataset Status (Phase 1: 45.2% Complete)

* **Total Images Extracted & Verified**: **22,600 frames** (Lossless PNG, 1920×1080)
* **Subjects Covered**: **97 unique subjects**
* **Metadata Coverage**: **100% tagged** in [`data/metadata.csv`](file:///e:/mineart-diffusion/data/metadata.csv)
* **Categories Included**:
  * **Dimensions**: Overworld, The Nether, The End
  * **Biomes & Landscapes**: Plains, Savanna/Acacia, Jungle, Dark Oak, Snow Forest, Mountains, Desert, Swamp, Cherry Blossom, Ocean & Coral Reefs
  * **Entities & Mobs**: Passive animals (cow, sheep, pig, chicken, bee, panda, camel, etc.), Hostile mobs (creeper, skeleton, zombie, spider, husk, warden, blaze, wither skeleton, brute piglin), Bosses (Ender Dragon, Wither), Villagers, Iron Golems
  * **Structures**: Villages (plains, acacia, desert), Pillager Outposts, Desert Temples, Bastion Remnants, Nether Fortresses, End Cities, Ancient Cities, Ruined Portals
  * **Items, Blocks & Nature**: Wood logs (all tree types), flora/flowers, vegetation, cacti, amethyst, magma blocks, diamond & netherite equipment

---

## Repository Structure

```text
mineart-diffusion/
│
├── data/
│   ├── images/               # Verified Minecraft frames (22,600 lossless PNGs)
│   ├── metadata.csv          # Complete dataset metadata with standardized tags
│   └── processed/            # Target directory for normalized training splits (train/val/test)
│
├── scripts/
│   ├── extract_frames.py     # Multi-threaded gameplay video frame extractor with auto-tagging
│   ├── watch_ss.py           # Real-time screenshot watcher from .minecraft
│   ├── prepare_dataset.py    # Automated dataset validation, cleaning & split pipeline
│   └── train_diffusion.py    # Custom PyTorch DDPM training script
│
├── frontend/                 # Interactive web interface for generation and exploration
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── notebooks/
│   └── dataset_analysis.ipynb # Interactive exploration and dataset analytics
│
├── requirements.txt          # Project dependencies
├── PROJECT_CONTEXT.md        # Comprehensive technical architecture & roadmap
├── README.md                 # Project documentation
└── .gitignore
```

---

## 🚀Active Workflow & Usage

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Dataset Collection & Frame Extraction

- **Batch Video Extraction**:
  Extract frames at optimal timestamps with automatic tagging:
  ```bash
  python scripts/extract_frames.py --tags "desert" "overworld"
  ```
- **Live Screenshot Watcher**:
  ```bash
  python scripts/watch_ss.py
  ```

### 3. Dataset Preprocessing & Cleaning (Upcoming)

Run the automated cleaning and split pipeline:

```bash
python scripts/prepare_dataset.py --input-dir data/images --output-dir data/processed --target-size 64
```

### 4. Custom Diffusion Model Training (Upcoming)

Train the custom DDPM model directly on the processed Minecraft dataset:

```bash
python scripts/train_diffusion.py --data-dir data/processed/train --epochs 100 --batch-size 32
```
