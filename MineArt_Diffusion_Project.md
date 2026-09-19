# MineArt Diffusion — Project Context & Architecture

## 1. Project Overview

**Project Name:** MineArt Diffusion  
**Core Mission:** Training a **custom Denoising Diffusion Probabilistic Model (DDPM)** from scratch to generate authentic, high-quality Minecraft-style artwork and imagery.

### Core Philosophy
- **No External Foundation Models**: We are intentionally NOT using pretrained SDXL, FLUX, or commercial image APIs.
- **Custom ML from Scratch**: The primary academic and engineering objective is building, training, and optimizing our own PyTorch DDPM architecture directly on the Minecraft domain.
- **Target Dataset**: **~50,000 Minecraft images** collected from in-game exploration, high-definition gameplay recordings, and authentic scene captures.
- **Minimal, Maintainable Architecture**: Clean modular pipeline: Raw Extraction → Structured Metadata → Preprocessing/Splits → Custom DDPM Training → Generation UI.

---

## 2. End-to-End Pipeline

```text
Minecraft Raw Captures (Target: ~50,000 | Current: 22,600)
                         ↓
             High-Speed Extraction & Tagging
         (120 FPS video parsing, metadata tagging)
                         ↓
               Dataset Preprocessing
   (Validation, deduplication, crop/scale, splits)
                         ↓
          Custom Diffusion Model (DDPM)
     (U-Net with Sinusoidal Positional Embeddings)
                         ↓
            Minecraft Artwork Generation
```

---

## 3. Current Dataset Status (Phase 1: 45.2% Complete)

* **Extracted & Verified Images**: **22,600 frames** (lossless PNG, 1920×1080)
* **Unique Subjects**: **97 subjects**
* **Metadata Status**: **100% tagged & audited** in `data/metadata.csv` (0 phantom records, standardized double quotes and comma spacing)

### Tagging Taxonomy & Distribution:
1. **Dimensions**: `"overworld"`, `"the nether"`, `"the end"`
2. **Biomes & Terrains**: Plains, Savanna/Acacia, Jungle, Dark Oak, Snow Forest, Mountains, Desert, Swamp, Cherry Blossom, Ocean, Coral Reefs
3. **Entities**: Passive mobs (cow, sheep, pig, chicken, bee, panda, camel, etc.), Hostile mobs (creeper, skeleton, zombie, spider, husk, warden, blaze, wither skeleton, brute piglin), Bosses (Ender Dragon, Wither), Villagers, Golems
4. **Structures**: Villages (plains, acacia, desert), Pillager Outposts, Desert Temples, Bastion Remnants, Nether Fortresses, End Cities, Ancient Cities, Ruined Portals
5. **Flora, Logs & Items**: Wood logs (oak, birch, spruce, dark oak, cherry, mangrove, pale oak, acacia, jungle), Flowers/Vegetation, Cacti, Magma blocks, Amethyst, Diamond/Netherite equipment

---

## 4. Phased Roadmap

### Phase 1: Dataset Collection & Engineering (CURRENT - Active)
- Collect raw in-game captures and gameplay video recordings into `data/images/` with automated metadata in `data/metadata.csv` (Target: ~50,000 images).
- High-speed frame extraction tool (`scripts/extract_frames.py`) with automatic frame budget calculation based on video duration.
- Automated validation: file existence checks, corrupt image detection, entropy filtering, and standardized tagging.

### Phase 2: Dataset Preprocessing Pipeline (NEXT)
- Develop `scripts/prepare_dataset.py` for automated preprocessing:
  - Aspect ratio handling / square center-crop
  - Resolution scaling (e.g., 64×64 / 128×128)
  - Train / Val / Test splitting (e.g., 80% / 10% / 10%)
  - Data normalization to `[-1, 1]`

### Phase 3: Custom Diffusion Model Development
- Develop and train custom PyTorch DDPM architecture on processed Minecraft images.
- Implement U-Net with Residual blocks, Attention mechanisms, and Sinusoidal Timestep Embeddings.
- Optimize variance schedules (linear vs. cosine beta schedules) and evaluate loss curves and FID scores.

### Phase 4: Frontend & Application Interface
- Complete the local interactive web UI (`frontend/`) for prompt-based / category-guided diffusion generation and exploration.

