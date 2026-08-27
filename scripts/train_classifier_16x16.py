"""
MineArt 16x16 Scene, Animal & Entity Tag Classifier
===================================================
A dual-head Convolutional Neural Network trained on 16x16 Minecraft images to:
  1. Multi-Class Subject Classification (101 classes: cow, sheep, pig, desert_village, mountains, etc.)
  2. Multi-Label Tag Classification (29 tags: overworld, entity, landscape, hostile, structure, block, etc.)

Features:
  - High-speed RAM caching (16x16 images take ~18MB RAM total, giving ~3-5 sec/epoch).
  - Dual-Head CNN with Residual Blocks & Global Average Pooling.
  - Automatic GPU (CUDA/AMP) & CPU fallback.
  - Generates loss/accuracy training curve plots and saves metadata/label mappings.
  - Includes full CLI for Training, Evaluation, and Single-Image Inference.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==============================================================================
# 1. Dataset with In-Memory Caching for Ultra-Fast Training
# ==============================================================================

class MineArt16x16Dataset(Dataset):
    """
    Loads Minecraft images from data/images, resizes them to 16x16,
    and extracts multi-class subject labels + multi-label tag vectors.
    """
    def __init__(
        self,
        metadata_csv: str = "data/metadata.csv",
        image_dir: str = "data/images",
        target_size: int = 16,
        cache_in_ram: bool = True,
        transform: transforms.Compose = None,
    ):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.transform = transform
        self.cache_in_ram = cache_in_ram

        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_csv}")

        print(f"[*] Reading metadata from {metadata_csv}...")
        self.df = pd.read_csv(metadata_csv)

        # 1. Subject mappings (Multi-Class: 101 subjects)
        self.subjects = sorted(self.df["subject"].dropna().unique().tolist())
        self.subject2idx = {subj: i for i, subj in enumerate(self.subjects)}
        self.idx2subject = {i: subj for i, subj in enumerate(self.subjects)}
        self.num_subjects = len(self.subjects)

        # 2. Tag mappings (Multi-Label: ~29 tags)
        raw_tags = set()
        for tag_str in self.df["tags"].dropna():
            for t in tag_str.split(","):
                clean = t.strip()
                if clean:
                    raw_tags.add(clean)
        self.tags = sorted(list(raw_tags))
        self.tag2idx = {tag: i for i, tag in enumerate(self.tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(self.tags)}
        self.num_tags = len(self.tags)

        print(f"[*] Dataset Statistics:")
        print(f"    - Total Rows: {len(self.df)}")
        print(f"    - Unique Subjects ({self.num_subjects}): {self.subjects[:10]} ...")
        print(f"    - Unique Tags ({self.num_tags}): {self.tags}")

        # Base transform for caching into (3, 16, 16) float tensors in [0, 1]
        self.base_resize = transforms.Compose([
            transforms.CenterCrop((1080, 1080)),  # Center crop 16:9 to square
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # Pre-process target vectors
        self.subject_labels = []
        self.tag_labels = []
        self.filenames = self.df["filename"].tolist()

        for _, row in self.df.iterrows():
            # Subject index
            subj = row["subject"]
            self.subject_labels.append(self.subject2idx.get(subj, 0))

            # Multi-hot tag binary vector
            tag_vec = torch.zeros(self.num_tags, dtype=torch.float32)
            if pd.notna(row["tags"]):
                for t in str(row["tags"]).split(","):
                    t_clean = t.strip()
                    if t_clean in self.tag2idx:
                        tag_vec[self.tag2idx[t_clean]] = 1.0
            self.tag_labels.append(tag_vec)

        self.subject_labels = torch.tensor(self.subject_labels, dtype=torch.long)
        self.tag_labels = torch.stack(self.tag_labels)

        # RAM Caching for 10x-50x training speedup
        self.cached_images = None
        if self.cache_in_ram:
            self._preload_images()

    def _preload_images(self):
        print(f"[*] Pre-loading and downscaling {len(self.filenames)} images to {self.target_size}x{self.target_size} into RAM...")
        t0 = time.time()
        tensors = []
        missing_count = 0
        blank_tensor = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)

        for fname in tqdm(self.filenames, desc="Caching 16x16 frames"):
            img_path = self.image_dir / fname
            if not img_path.exists():
                missing_count += 1
                tensors.append(blank_tensor)
                continue

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    tensor = self.base_resize(img)
                    tensors.append(tensor)
            except Exception as e:
                missing_count += 1
                tensors.append(blank_tensor)

        self.cached_images = torch.stack(tensors)  # Shape: (N, 3, 16, 16)
        ram_mb = (self.cached_images.element_size() * self.cached_images.nelement()) / (1024 * 1024)
        elapsed = time.time() - t0
        print(f"[✓] Caching completed in {elapsed:.2f}s! Total RAM used: {ram_mb:.2f} MB. (Missing: {missing_count})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_images is not None:
            image = self.cached_images[idx]
        else:
            fname = self.filenames[idx]
            img_path = self.image_dir / fname
            if img_path.exists():
                with Image.open(img_path) as img:
                    image = self.base_resize(img.convert("RGB"))
            else:
                image = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        # Return (image_tensor, subject_idx, tag_multihot_vector)
        return image, self.subject_labels[idx], self.tag_labels[idx]

    def save_labels_json(self, filepath: str = "checkpoints/classifier_16x16_labels.json"):
        """Saves label dictionaries for inference and web app integration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "target_size": self.target_size,
            "num_subjects": self.num_subjects,
            "subjects": self.subjects,
            "subject2idx": self.subject2idx,
            "num_tags": self.num_tags,
            "tags": self.tags,
            "tag2idx": self.tag2idx
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] Saved label mappings to: {filepath}")


# ==============================================================================
# 2. MineArt Dual-Head 16x16 ResNet Architecture
# ==============================================================================

class ResidualBlock(nn.Module):
    """Residual convolutional block designed for compact 16x16 feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class MineArt16x16Net(nn.Module):
    """
    Dual-Head CNN for 16x16 Minecraft Images:
      - Head 1: Multi-Class Subject Output (logits for 101 subjects)
      - Head 2: Multi-Label Tag Output (logits for 29 binary tags)
    """
    def __init__(self, num_subjects: int = 101, num_tags: int = 29, base_dim: int = 48):
        super().__init__()
        # Stem: (3, 16, 16) -> (base_dim, 16, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )

        # Stage 1: (base_dim, 16, 16)
        self.stage1 = nn.Sequential(
            ResidualBlock(base_dim),
            ResidualBlock(base_dim),
        )

        # Stage 2: (base_dim, 16, 16) -> (base_dim*2, 8, 8)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(base_dim * 2),
            ResidualBlock(base_dim * 2),
        )

        # Stage 3: (base_dim*2, 8, 8) -> (base_dim*4, 4, 4)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 4),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(base_dim * 4),
        )

        # Global Pooling -> Feature Embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = base_dim * 4
        self.dropout = nn.Dropout(p=0.25)

        # Dual Heads
        self.subject_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_subjects),
        )

        self.tag_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_tags),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        feat = self.global_pool(x).flatten(1)
        feat = self.dropout(feat)

        subject_logits = self.subject_head(feat)
        tag_logits = self.tag_head(feat)
        return subject_logits, tag_logits


# ==============================================================================
# 3. Training & Evaluation Engine
# ==============================================================================

def compute_metrics(
    subject_logits: torch.Tensor,
    subject_targets: torch.Tensor,
    tag_logits: torch.Tensor,
    tag_targets: torch.Tensor,
    tag_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes Top-1 subject accuracy and Multi-Label Tag F1/Precision/Recall."""
    # 1. Subject Top-1 Accuracy
    _, preds = torch.max(subject_logits, 1)
    subject_acc = (preds == subject_targets).float().mean().item()

    # 2. Multi-Label Tag Metrics (Sigmoid > 0.5)
    tag_probs = torch.sigmoid(tag_logits)
    tag_preds = (tag_probs >= tag_threshold).float()

    tp = (tag_preds * tag_targets).sum().item()
    fp = (tag_preds * (1 - tag_targets)).sum().item()
    fn = ((1 - tag_preds) * tag_targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    tag_acc = ((tag_preds == tag_targets).float().mean()).item()

    return {
        "subject_acc": subject_acc,
        "tag_acc": tag_acc,
        "tag_f1": f1,
        "tag_precision": precision,
        "tag_recall": recall,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    subject_criterion: nn.Module,
    tag_criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
    subject_weight: float = 1.0,
    tag_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_subj_loss = 0.0
    total_tag_loss = 0.0
    all_metrics = []

    for images, subjects, tags in loader:
        images = images.to(device, non_blocking=True)
        subjects = subjects.to(device, non_blocking=True)
        tags = tags.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                subj_logits, tag_logits = model(images)
                loss_s = subject_criterion(subj_logits, subjects)
                loss_t = tag_criterion(tag_logits, tags)
                loss = (subject_weight * loss_s) + (tag_weight * loss_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            subj_logits, tag_logits = model(images)
            loss_s = subject_criterion(subj_logits, subjects)
            loss_t = tag_criterion(tag_logits, tags)
            loss = (subject_weight * loss_s) + (tag_weight * loss_t)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_subj_loss += loss_s.item() * images.size(0)
        total_tag_loss += loss_t.item() * images.size(0)

        with torch.no_grad():
            m = compute_metrics(subj_logits, subjects, tag_logits, tags)
            all_metrics.append(m)

    n_samples = len(loader.dataset)
    avg_m = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg_m["loss"] = total_loss / n_samples
    avg_m["loss_subject"] = total_subj_loss / n_samples
    avg_m["loss_tags"] = total_tag_loss / n_samples
    return avg_m


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    subject_criterion: nn.Module,
    tag_criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for images, subjects, tags in loader:
            images = images.to(device, non_blocking=True)
            subjects = subjects.to(device, non_blocking=True)
            tags = tags.to(device, non_blocking=True)

            subj_logits, tag_logits = model(images)
            loss_s = subject_criterion(subj_logits, subjects)
            loss_t = tag_criterion(tag_logits, tags)
            loss = loss_s + loss_t

            total_loss += loss.item() * images.size(0)
            m = compute_metrics(subj_logits, subjects, tag_logits, tags)
            all_metrics.append(m)

    n_samples = len(loader.dataset)
    avg_m = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg_m["loss"] = total_loss / n_samples
    return avg_m


# ==============================================================================
# 4. Training Plotting & Artifact Generation
# ==============================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: str = "checkpoints/training_history_16x16.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#ff79c6", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#8be9fd", lw=2)
    axes[0].set_title("Total Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subject Accuracy
    axes[1].plot(epochs, history["train_subj_acc"], label="Train Subj Acc", color="#50fa7b", lw=2)
    axes[1].plot(epochs, history["val_subj_acc"], label="Val Subj Acc", color="#f1fa8c", lw=2)
    axes[1].set_title("Subject Accuracy (101 Classes)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Tag F1 Score
    axes[2].plot(epochs, history["train_tag_f1"], label="Train Tag F1", color="#bd93f9", lw=2)
    axes[2].plot(epochs, history["val_tag_f1"], label="Val Tag F1", color="#ffb86c", lw=2)
    axes[2].set_title("Multi-Label Tag F1 Score (29 Tags)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[✓] Training curves saved to: {save_path}")


# ==============================================================================
# 5. Main Training Pipeline
# ==============================================================================

def train_classifier(
    metadata_csv: str = "data/metadata.csv",
    image_dir: str = "data/images",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_dir: str = "checkpoints",
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Device selection (CUDA with Ampere FP16 or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Dataset & Caching
    dataset = MineArt16x16Dataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        target_size=16,
        cache_in_ram=True,
    )
    dataset.save_labels_json(str(save_path / "classifier_16x16_labels.json"))

    # Train / Val / Test Split (80% / 10% / 10%)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    print(f"[*] Dataset Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Model & Optimizers
    model = MineArt16x16Net(
        num_subjects=dataset.num_subjects,
        num_tags=dataset.num_tags,
        base_dim=48,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Architecture Initialized! Trainable Parameters: {total_params:,}")

    # Loss Functions:
    # 1. Subject: Multi-Class Cross Entropy with label smoothing
    subject_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 2. Tags: Multi-Label Binary Cross Entropy with Logits
    tag_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Tracking History
    history = {
        "train_loss": [], "val_loss": [],
        "train_subj_acc": [], "val_subj_acc": [],
        "train_tag_f1": [], "val_tag_f1": [],
    }

    best_val_score = -1.0
    best_model_path = save_path / "best_classifier_16x16.pt"

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Subj Acc (Val)':^14} | {'Tag F1 (Val)':^12} | {'Time':^8}")
    print("=" * 80)

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            subject_criterion=subject_criterion,
            tag_criterion=tag_criterion,
            device=device,
            scaler=scaler,
        )

        val_res = evaluate(
            model=model,
            loader=val_loader,
            subject_criterion=subject_criterion,
            tag_criterion=tag_criterion,
            device=device,
        )

        scheduler.step()
        epoch_time = time.time() - t0

        # Record history
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["train_subj_acc"].append(train_res["subject_acc"])
        history["val_subj_acc"].append(val_res["subject_acc"])
        history["train_tag_f1"].append(train_res["tag_f1"])
        history["val_tag_f1"].append(val_res["tag_f1"])

        # Composite score to track best checkpoint (50% Subject Acc + 50% Tag F1)
        val_score = (val_res["subject_acc"] + val_res["tag_f1"]) / 2.0

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_res,
                "num_subjects": dataset.num_subjects,
                "num_tags": dataset.num_tags,
                "subjects": dataset.subjects,
                "tags": dataset.tags,
            }, best_model_path)

        mark = " ★ BEST" if is_best else ""
        print(
            f"{epoch:^7d} | "
            f"{train_res['loss']:^10.4f} | "
            f"{val_res['loss']:^10.4f} | "
            f"{val_res['subject_acc']*100:^12.2f}% | "
            f"{val_res['tag_f1']*100:^10.2f}% | "
            f"{epoch_time:^6.1f}s{mark}"
        )

    total_time = time.time() - start_train_time
    print("=" * 80)
    print(f"[✓] Training Completed in {total_time/60:.2f} minutes!")

    # Plot training curves
    plot_training_history(history, str(save_path / "classifier_16x16_training_plot.png"))

    # Final Evaluation on Test Split
    print("\n[*] Evaluating Best Model on Held-out Test Set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_res = evaluate(model, test_loader, subject_criterion, tag_criterion, device)
    print("=" * 50)
    print(f"   TEST SET EVALUATION RESULTS:")
    print(f"   - Subject Accuracy (101 Classes): {test_res['subject_acc']*100:.2f}%")
    print(f"   - Multi-Label Tag Accuracy:       {test_res['tag_acc']*100:.2f}%")
    print(f"   - Multi-Label Tag F1 Score:       {test_res['tag_f1']*100:.2f}%")
    print(f"   - Multi-Label Tag Precision:      {test_res['tag_precision']*100:.2f}%")
    print(f"   - Multi-Label Tag Recall:         {test_res['tag_recall']*100:.2f}%")
    print("=" * 50)


# ==============================================================================
# 6. Single-Image Inference
# ==============================================================================

def predict_image(image_path: str, checkpoint_path: str = "checkpoints/best_classifier_16x16.pt"):
    """Runs classification on a single Minecraft image at 16x16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        print(f"[-] Checkpoint not found at: {checkpoint_path}. Train the model first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    subjects = checkpoint["subjects"]
    tags = checkpoint["tags"]

    model = MineArt16x16Net(num_subjects=len(subjects), num_tags=len(tags)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocess image to 16x16
    preprocess = transforms.Compose([
        transforms.CenterCrop((1080, 1080)) if Image.open(image_path).size[0] > 1080 else transforms.Resize((16, 16)),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        subj_logits, tag_logits = model(tensor)

        # Subject Top-5
        subj_probs = torch.softmax(subj_logits, dim=-1)[0]
        top5_probs, top5_indices = torch.topk(subj_probs, 5)

        # Tags > 0.4
        tag_probs = torch.sigmoid(tag_logits)[0]
        active_tags = []
        for i, p in enumerate(tag_probs):
            if p >= 0.35:
                active_tags.append((tags[i], p.item()))
        active_tags.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 50)
    print(f"🎮 MineArt 16x16 Classifier Prediction for: {image_path}")
    print("=" * 50)
    print(f"🏆 Top Predicted Subject: {subjects[top5_indices[0]]} ({top5_probs[0]*100:.1f}%)")
    print("\nTop 5 Subject Candidates:")
    for prob, idx in zip(top5_probs, top5_indices):
        print(f"   • {subjects[idx]:<25} : {prob*100:.1f}%")

    print("\n🏷️ Predicted Tags (Multi-Label):")
    if active_tags:
        for tag_name, score in active_tags:
            print(f"   • {tag_name:<20} : {score*100:.1f}%")
    else:
        print("   • (No tags above threshold)")
    print("=" * 50)


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineArt 16x16 Classifier Training & Inference")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--predict", type=str, default=None, help="Path to image for inference")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--images", type=str, default="data/images", help="Path to raw image folder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_classifier_16x16.pt", help="Checkpoint path")

    args = parser.parse_args()

    if args.predict:
        predict_image(args.predict, checkpoint_path=args.checkpoint)
    else:
        train_classifier(
            metadata_csv=args.metadata,
            image_dir=args.images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
