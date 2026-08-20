# MineArt Diffusion — Project Context & Development Plan

## 1. Project Overview

**Project Name:** MineArt Diffusion

MineArt Diffusion is a machine-learning-focused generative and analytical system for creating Minecraft-style artwork.

The user can provide:
1. A text prompt
2. An input image
3. Optionally, an input image + text instruction

The system generates Minecraft-style artwork using a trained diffusion model, analyzes the generated image with computer vision, visualizes the results, and converts the final artwork into a Minecraft-compatible poster/painting asset.

The core academic focus MUST be:
- Machine Learning
- Deep Learning
- Diffusion Models
- Model Training
- Fine-tuning
- Computer Vision
- Data Visualization
- Applied Mathematics
- Dataset Engineering
- Model Evaluation

The website is the product interface around the ML system. Do NOT make this merely a wrapper around an external image-generation API.

---

# 2. Core User Flow

```text
                         USER
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Upload Image                Enter Prompt
             |                           |
             +-------------+-------------+
                           |
                           v
                  Input Processing
                           |
                           v
                  Conditioning Layer
                           |
                           v
                  MineArt Diffusion
                           |
                           v
                  Generated Artwork
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Image Analysis             Generation Metrics
             |                           |
             +-------------+-------------+
                           |
                           v
                  Visualization Layer
                           |
                           v
                  Minecraft Optimizer
                           |
                           v
                 Minecraft Poster/Asset
                           |
                           v
                         Export
```

---

# 3. Generation Modes

## Text → Image

Example:
> A medieval Minecraft castle surrounded by snowy mountains at sunset.

Pipeline:

```text
Text Prompt → Text Conditioning → Diffusion → Minecraft Artwork
```

## Image → Image

```text
Input Image → Image Conditioning → MineArt Diffusion → Minecraft Artwork
```

The model should preserve useful structural information while transforming the visual style.

## Image + Text → Image

Example:

```text
Input Image
+
"Make this a dark fantasy Minecraft painting"
        ↓
Combined Conditioning
        ↓
Diffusion Model
        ↓
Minecraft Artwork
```

This is an advanced feature and should be implemented after the basic generation pipeline works.

---

# 4. ML Architecture

MineArt has three major ML components:

```text
                    MINEART ML
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   GENERATION       ANALYSIS        EVALUATION
        |               |               |
        v               v               v
   Diffusion       Segmentation      Metrics
     Model            Model
```

## Model A — MineArt-DDPM

Build a smaller diffusion model **from scratch using PyTorch**.

Purpose:
- Demonstrate understanding of diffusion mathematics
- Implement forward diffusion
- Implement noise schedules
- Implement reverse denoising
- Implement timestep embeddings
- Train a U-Net
- Run controlled experiments
- Produce a genuinely custom-trained model

Initial target:
- 64×64 images
- DDPM
- U-Net
- Initially 1000 diffusion timesteps

This is primarily the academic/research model.

## Model B — MineArt-SD

Use a strong pretrained diffusion model and fine-tune it for Minecraft imagery using LoRA.

Technology:
- PyTorch
- Hugging Face Diffusers
- PEFT/LoRA

The project should compare:

```text
Custom DDPM trained from scratch
              VS
Pretrained diffusion model + Minecraft LoRA
```

This comparison is a major ML experiment.

---

# 5. Diffusion Mathematics

The implementation and documentation must explicitly cover the mathematics.

Forward diffusion:

q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)x_{t-1}, beta_t I)

Define:

alpha_t = 1 - beta_t

and:

alpha_bar_t = product(alpha_s), for s = 1 ... t

Direct noisy-image sampling:

x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon

where:

epsilon ~ N(0,I)

Noise prediction:

epsilon_theta(x_t,t)

Training objective:

L = E[ ||epsilon - epsilon_theta(x_t,t)||² ]

These equations should be reflected in both the implementation and technical documentation.

---

# 6. Noise Schedule Experiments

Experiment with at least:
1. Linear beta schedule
2. Cosine beta schedule

Potentially investigate another schedule later.

Visualize:
- beta(t)
- alpha(t)
- cumulative alpha-bar
- signal-to-noise ratio
- training loss
- generation quality

Example:

```text
Experiment A → Linear schedule
Experiment B → Cosine schedule
```

Compare their effect on training and generation.

---

# 7. Dataset Strategy

The dataset should be **created by the team**, primarily from thousands of Minecraft screenshots captured in-game.

Do not rely entirely on a generic downloaded dataset.

Raw screenshots should be placed into one folder:

```text
minecraft_raw/
    screenshot1.png
    screenshot2.png
    random_name.jpg
    Screenshot_2026_01.png
```

A Python dataset-preparation script must automatically:

1. Find images recursively
2. Validate image files
3. Remove or report corrupt files
4. Detect duplicates where practical
5. Record original dimensions
6. Convert images to a consistent format
7. Resize/crop to training resolution
8. Rename files consistently
9. Split into train/validation/test
10. Generate metadata
11. Produce dataset statistics

The user must NOT manually rename thousands of screenshots.

Target structure:

```text
minecraft_dataset/
├── train/
│   └── images/
├── validation/
│   └── images/
├── test/
│   └── images/
├── metadata.jsonl
└── dataset_info.json
```

---

# 8. Dataset Variety

The screenshots should intentionally contain substantial visual diversity.

Possible categories:

```text
Landscapes
    - plains
    - forests
    - mountains
    - deserts
    - oceans
    - snowy environments

Structures
    - houses
    - villages
    - castles
    - bridges
    - cities
    - towers

Environments
    - caves
    - Nether
    - End
    - underwater

Objects
    - farms
    - redstone builds
    - decorations
    - vehicles

Characters
    - player
    - mobs
    - groups
```

Exact categories can evolve based on the collected dataset.

---

# 9. Dataset Visualization

Before training, generate real dataset analytics:
- Total image count
- Train/validation/test split
- Image resolution distribution
- Aspect ratio distribution
- Category distribution
- Dominant color distribution
- Duplicate count
- File format distribution
- Sample image grids

This is part of the Data Visualization requirement, not decoration.

---

# 10. Data Augmentation Experiments

Investigate whether augmentation improves the model.

Possible experiments:

```text
A: No augmentation
B: Horizontal flip
C: Flip + crop
D: Flip + crop + carefully selected color augmentation
```

Compare metrics and generated samples.

Avoid transformations that destroy important Minecraft visual structure.

---

# 11. Experiment Tracking

Every serious training run must be reproducible.

Track:
- Experiment ID
- Model version
- Dataset version
- Number of images
- Image resolution
- Batch size
- Learning rate
- Optimizer
- Scheduler
- Epochs
- Diffusion timesteps
- Noise schedule
- LoRA configuration
- Random seed
- Training loss
- Validation loss
- FID
- KID
- CLIP score
- Inference time
- GPU information

Recommended tool: **MLflow**

Never hard-code fake metrics. Every displayed metric must come from a real experiment.

---

# 12. Computer Vision Analysis

After generation, MineArt must analyze the visual composition of the image.

Pipeline:

```text
Generated Image
      ↓
Semantic Segmentation
      ↓
Pixel-level class prediction
      ↓
Class pixel counts
      ↓
Percentage calculation
      ↓
Visualization
```

Possible classes:

```text
sky
cloud
land
water
mountain
tree
vegetation
building
character
road
other
```

The final class list can be adjusted according to the actual dataset.

Example result:

```text
Sky          24.6%
Land         21.4%
Mountain     18.7%
Building     12.1%
Trees         9.2%
Clouds        8.3%
Other         5.7%
```

---

# 13. Segmentation Model

Investigate:
- SegFormer
- DeepLabV3+
- U-Net

Eventually fine-tune a segmentation model using a small manually labeled Minecraft dataset.

Example:

```text
image_001.png
mask_001.png

image_002.png
mask_002.png
```

The masks encode semantic classes.

This provides another genuine ML training task.

---

# 14. Segmentation Mathematics

For each pixel:

f(x,y) → c

where c is a semantic class.

For class c:

P(c) = (N_c / N_total) × 100

where:
- N_c = pixels belonging to class c
- N_total = total pixels

This mathematical calculation powers the composition visualization.

---

# 15. Segmentation Evaluation

Use quantitative metrics.

At minimum investigate:

Intersection over Union:

IoU = TP / (TP + FP + FN)

Calculate IoU per class.

Do not use example values as real results. All final scores must be experimentally measured.

---

# 16. Generated Image Evaluation

Use quantitative metrics where appropriate:

### FID
Fréchet Inception Distance.

### KID
Kernel Inception Distance.

### CLIP similarity
Measures semantic alignment between prompt and generated image.

### Diversity
Measures whether different seeds/prompts produce meaningfully different outputs.

### Optional metrics
MSE / SSIM can be used for suitable image-to-image comparisons.

Metrics must be interpreted according to their actual purpose.

---

# 17. Data Visualization System

Create a dedicated **Diffusion Lab / Analytics** area.

Visualize:

### Training
- Training loss
- Validation loss
- Learning rate
- Epoch progress

### Diffusion
- Forward diffusion
- Reverse diffusion
- Noise schedule
- Sampling steps

### Dataset
- Dataset size
- Categories
- Resolution
- Color distribution
- Split distribution

### Evaluation
- FID
- KID
- CLIP
- Segmentation IoU
- Diversity
- Generation time

### Image analysis
- Semantic segmentation map
- Composition percentages
- Object counts
- Color distribution

Use interactive web charts rather than static images wherever practical.

---

# 18. Diffusion Explorer

Create an interactive view of the diffusion process.

Forward:

```text
Original
   ↓
t = 100
   ↓
t = 300
   ↓
t = 500
   ↓
t = 700
   ↓
t = 900
   ↓
Noise
```

Reverse:

```text
Noise
   ↓
Step 900
   ↓
Step 700
   ↓
Step 500
   ↓
Step 300
   ↓
Step 100
   ↓
Generated Image
```

Provide a timestep/denoising slider where practical.

The visualization must show the actual saved/intermediate model output, not fake animations.

---

# 19. Latent/Embedding Visualization

Advanced feature:

Extract image embeddings and visualize using:
- PCA
- t-SNE
- UMAP

Possible clusters:
- landscapes
- architecture
- characters
- environments
- objects

This combines ML, dimensionality reduction, mathematics, and visualization.

---

# 20. Generation Feedback Loop

Advanced feature:

Compare requested composition with generated composition.

Example:

```text
Requested:
Mountain > 30%
Sky < 25%

Actual:
Mountain = 21%
Sky = 38%
```

The system can report a composition mismatch.

Eventually:

```text
Prompt
  ↓
Generate
  ↓
Segment
  ↓
Measure
  ↓
Check constraints
  |
  +---- NO ----> Regenerate / Adjust
  |
  +---- YES ---> Final Result
```

This should only be implemented after the base generation and segmentation pipelines work.

---

# 21. Minecraft Export

The final image passes through:

```text
Generated Image
      ↓
Resolution processing
      ↓
Aspect ratio processing
      ↓
Color quantization
      ↓
Minecraft palette optimization
      ↓
Quality check
      ↓
Minecraft asset/resource pack generation
```

Applied mathematics can be used for palette optimization.

For each pixel p and palette P:

c* = argmin(d(p,c)), c ∈ P

Investigate:
- Euclidean RGB distance
- K-means quantization
- Median-cut
- Other appropriate methods

Compare methods experimentally if time allows.

---

# 22. Full-Stack Architecture

```text
React + TypeScript
        |
        | REST / WebSocket
        v
FastAPI
        |
   +----+----+
   |         |
   v         v
Redis      PostgreSQL
   |
   v
Celery GPU Workers
   |
   +-----------------------------+
   |             |               |
   v             v               v
Diffusion   Segmentation     Evaluation
   |
   v
Image Processing
   |
   v
Minecraft Export
```

The frontend is an interface to the ML system.

---

# 23. Frontend

Use:
- React
- TypeScript
- Vite
- Tailwind CSS
- Apache ECharts

Suggested sections:

```text
Dashboard
Generate
Gallery
Image Analysis
Diffusion Lab
Experiments
Dataset Analytics
Model Comparison
Minecraft Export
```

The interface should look like a serious ML/research product, not a generic AI image generator.

---

# 24. Backend

Use **FastAPI**.

Suggested structure:

```text
backend/
├── app/
│   ├── api/
│   │   ├── generation.py
│   │   ├── analysis.py
│   │   ├── experiments.py
│   │   ├── models.py
│   │   └── export.py
│   ├── services/
│   │   ├── generation_service.py
│   │   ├── segmentation_service.py
│   │   ├── evaluation_service.py
│   │   ├── optimization_service.py
│   │   └── export_service.py
│   └── workers/
│       └── generation_worker.py
```

Do not put heavy ML logic directly in API route functions.

---

# 25. Database

Use **PostgreSQL**.

Store:
- users
- generations
- prompts
- models
- model versions
- experiments
- metrics
- segmentation results
- export records

Do not store large image files directly in PostgreSQL.

Use object storage such as **MinIO/S3-compatible storage**.

---

# 26. Background Processing

Diffusion and segmentation can be expensive.

Use:

- Redis
- Celery

Workflow:

```text
Browser
   ↓
POST /generate
   ↓
Create job
   ↓
Redis
   ↓
Celery GPU worker
   ↓
Diffusion
   ↓
Segmentation
   ↓
Evaluation
   ↓
Save results
   ↓
Frontend receives status/result
```

Use WebSockets for live generation/training progress where useful.

---

# 27. ML Technology Stack

```text
Python
PyTorch
Hugging Face Diffusers
PEFT / LoRA
TorchMetrics or appropriate metric implementations
OpenCV
Pillow
NumPy
Pandas
scikit-learn
MLflow
CUDA
```

Important: implement the educationally important parts of the custom DDPM yourself instead of hiding the entire algorithm behind a framework.

---

# 28. Web Technology Stack

```text
Frontend:
React
TypeScript
Vite
Tailwind CSS
Apache ECharts

Backend:
FastAPI
Pydantic

Database:
PostgreSQL

Queue:
Redis
Celery

Storage:
MinIO / S3-compatible storage

Deployment:
Docker
Docker Compose

Version Control:
Git
GitHub
```

---

# 29. Repository Structure

```text
MineArt-Diffusion/
│
├── README.md
├── PROJECT_CONTEXT.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── frontend/
├── backend/
│
├── ml/
│   ├── dataset/
│   ├── diffusion/
│   │   ├── ddpm/
│   │   ├── unet/
│   │   ├── schedules/
│   │   └── sampling/
│   ├── finetuning/
│   ├── segmentation/
│   ├── evaluation/
│   └── inference/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── experiments/
│   ├── configs/
│   ├── results/
│   └── reports/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── train_ddpm.py
│   ├── train_segmentation.py
│   └── evaluate_model.py
│
├── notebooks/
│   ├── dataset_analysis/
│   ├── diffusion_analysis/
│   └── model_evaluation/
│
└── docs/
    ├── architecture.md
    ├── mathematics.md
    ├── training.md
    ├── experiments.md
    └── dataset.md
```

---

# 30. Development Phases

## Phase 1 — Dataset

Build:
- Screenshot collection workflow
- Automated dataset organizer
- Validation
- Preprocessing
- Train/validation/test split
- Metadata
- Dataset visualization

Deliverable:
> A clean, reproducible Minecraft dataset.

## Phase 2 — Custom DDPM

Implement:
- Beta schedules
- Forward diffusion
- Timestep embeddings
- U-Net
- Noise prediction
- MSE loss
- Reverse sampling
- Training loop
- Checkpoints

Deliverable:
> A diffusion model trained from scratch.

## Phase 3 — Diffusion Experiments

Experiment with:
- Linear vs cosine schedules
- Dataset sizes
- Augmentation
- Learning rates
- Batch sizes
- Epochs
- Sampling steps

Track everything with MLflow.

Deliverable:
> Experimental evidence showing how training choices affect results.

## Phase 4 — Pretrained Fine-Tuning

Implement:
- Pretrained diffusion model
- Minecraft LoRA
- Training configuration
- Inference
- Model versioning

Deliverable:
> High-quality Minecraft-specialized diffusion model.

## Phase 5 — Image Conditioning

Implement image-to-image generation.

Then:
```text
Image + Text
```

Deliverable:
> Multimodal MineArt generation.

## Phase 6 — Segmentation

Implement:
- Segmentation model
- Minecraft labels
- Small labeled dataset
- Fine-tuning
- Pixel-level prediction
- Class percentages
- IoU evaluation

Deliverable:
> Automatic visual composition analysis.

## Phase 7 — Analytics

Build:
- Training charts
- Diffusion explorer
- Dataset dashboard
- Model comparison
- Image composition charts
- Segmentation visualization
- Evaluation dashboard

Deliverable:
> A serious ML visualization dashboard.

## Phase 8 — Minecraft Export

Implement:
- Resizing
- Aspect ratio handling
- Palette optimization
- Minecraft asset generation
- Export

Deliverable:
> Generated Minecraft-compatible artwork.

## Phase 9 — Full-Stack Integration

Connect:

```text
React
 ↓
FastAPI
 ↓
Redis/Celery
 ↓
GPU ML Services
 ↓
PostgreSQL + MinIO
```

Deliverable:
> Complete MineArt application.

## Phase 10 — Testing & Research Documentation

Document:
- Dataset methodology
- Mathematical formulation
- Architecture
- Training experiments
- Hyperparameters
- Evaluation
- Limitations
- Conclusions

Use actual graphs and experimental results.

---

# 31. Minimum Serious Version

If the full scope becomes too large, prioritize:

```text
Custom DDPM
       +
Minecraft Dataset
       +
Training
       +
Evaluation
       +
Text Generation
       +
Segmentation Analysis
       +
Data Visualization
       +
Web Interface
       +
Minecraft Export
```

Image-to-image and feedback-based regeneration are advanced features and can come later.

---

# 32. What NOT to Do

Do NOT:
- Build a simple wrapper around an external image-generation API.
- Claim a pretrained model is your own trained model.
- Hard-code fake ML metrics.
- Generate fake training graphs.
- Use placeholder FID/CLIP/IoU values in the final system.
- Put all ML code inside FastAPI routes.
- Store huge image files directly in PostgreSQL.
- Manually rename thousands of dataset images.
- Start with the website before understanding the ML pipeline.
- Try to train a full Stable-Diffusion-class model from scratch.
- Add dozens of features before the core diffusion training works.

---

# 33. Definition of Success

MineArt Diffusion should eventually allow a user to:

1. Enter a text prompt or upload an image.
2. Generate Minecraft-style artwork using our trained model.
3. View generation progress/intermediate diffusion states where practical.
4. View quantitative model/image metrics.
5. See semantic segmentation of the generated image.
6. See the percentage of the image occupied by different visual classes.
7. Explore dataset/model/training visualizations.
8. Compare different trained model versions.
9. Optimize the artwork for Minecraft.
10. Export the final artwork as a Minecraft-compatible poster/painting asset.

The team should be able to explain:

```text
Dataset
   ↓
Preprocessing
   ↓
Mathematics
   ↓
Diffusion
   ↓
Architecture
   ↓
Training
   ↓
Experiments
   ↓
Evaluation
   ↓
Visualization
   ↓
Application
```

This chain is the academic backbone of MineArt Diffusion.

---

# 34. Guiding Principle for Antigravity IDE

Prioritize the ML/research pipeline over superficial UI features.

Correct order:

```text
UNDERSTAND
    ↓
BUILD DATASET
    ↓
TRAIN
    ↓
MEASURE
    ↓
VISUALIZE
    ↓
IMPROVE MODEL
    ↓
BUILD API
    ↓
BUILD FRONTEND
    ↓
INTEGRATE
    ↓
DEPLOY
```

Do not prematurely build a polished dashboard around a model that has not been trained and evaluated.

Every important ML result must come from actual experiments.

Every visualization must be generated from real data.

Every model version must be reproducible.

The final application should demonstrate that MineArt Diffusion is a **real machine-learning project with a software product around it**, not a software product that merely calls an AI model.
