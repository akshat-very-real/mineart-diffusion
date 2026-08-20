# MineArt Diffusion

MineArt Diffusion is a machine-learning project aimed at generating Minecraft-style artwork and paintings using diffusion models.

## Phase 1: Dataset Creation & Preprocessing

Phase 1 provides an automated, reproducible data pipeline that converts raw in-game Minecraft screenshots into a clean, normalized, ML-ready dataset.

### Project Structure (Phase 1)

```
MineArt-Diffusion/
│
├── data/
│   ├── raw/                  # Place raw screenshots here (.png, .jpg, etc.)
│   └── processed/            # Generated clean dataset
│       ├── train/            # Training split (e.g. 80%)
│       ├── validation/       # Validation split (e.g. 10%)
│       ├── test/             # Test split (e.g. 10%)
│       ├── reports/          # Visual summary plots and sample contact sheet
│       ├── dataset_metadata.csv
│       └── dataset_statistics.json
│
├── scripts/
│   └── prepare_dataset.py    # Main CLI preprocessing pipeline
│
├── notebooks/
│   └── dataset_analysis.ipynb # Interactive exploration and quality analysis
│
├── README.md
├── PROJECT_CONTEXT.md
└── requirements.txt
```

---

### Getting Started

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Add Raw Screenshots
Copy your Minecraft screenshots into the `data/raw/` directory.

#### 3. Run Preprocessing Pipeline
```bash
python scripts/prepare_dataset.py
```

##### Optional CLI Arguments:
- `--input`: Path to raw screenshots folder (default: `data/raw`)
- `--output`: Path to processed destination (default: `data/processed`)
- `--size`: Target square resolution in pixels (default: `64` for 64x64)
- `--train-ratio`: Proportion for training split (default: `0.8`)
- `--val-ratio`: Proportion for validation split (default: `0.1`)
- `--test-ratio`: Proportion for test split (default: `0.1`)
- `--seed`: Random seed for reproducible splitting (default: `42`)
- `--no-plots`: Disable automatic visual report generation

Example with custom parameters:
```bash
python scripts/prepare_dataset.py --input data/raw --output data/processed --size 64 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

---

### Pipeline Capabilities
- **Recursive Scanning**: Detects `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp` files.
- **Integrity Validation**: Catches and logs corrupt or unreadable image files without crashing.
- **Deduplication**: Calculates MD5 checksums to detect and separate duplicate screenshots.
- **Aspect Ratio Normalization**: Center-crops images to square aspect ratio (without distortion) and resizes to target resolution (64x64) using Lanczos resampling.
- **Color Normalization**: Converts all images to standard 3-channel RGB PNG format.
- **Split Management**: Partition dataset deterministically into `train/`, `validation/`, and `test/` splits with clean sequential filenames (`train_00001.png`, etc.).
- **Metadata & Statistics**: Exports `dataset_metadata.csv` and `dataset_statistics.json`.
- **Visual Reports**: Generates distribution graphs and a sample contact sheet in `data/processed/reports/`.

---

### Dataset Analysis Notebook
Launch Jupyter to explore dataset metrics, inspect distributions, and verify training readiness:
```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```
