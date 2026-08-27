"""MineArt 16x16 Scene, Animal, Entity and Tag Classification Model.

This module implements a dual-head Convolutional Neural Network with residual
connections designed for high-speed multi-task classification on 16x16 Minecraft
pixel art representations.

Tasks Performed:
1. Multi-Class Subject Classification: Classifies distinct Minecraft mob identities,
   biomes, blocks, and structures across 101 target classes using Cross-Entropy.
2. Multi-Label Tag Classification: Simultaneously predicts 29 environmental,
   dimensional, and entity tags (e.g. overworld, hostile, structure, nether, cave)
   using Binary Cross-Entropy with Logits.
3. Fast RAM Caching: Pre-processes and holds the entire 16x16 dataset in memory
   to eliminate disk bottlenecks and maximize training throughput.
"""

import argparse
import json
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
# pyrefly: ignore
from torchvision import transforms
from tqdm import tqdm


class MineArt16x16Dataset(Dataset):
    """Dataset loader with in-memory caching for 16x16 classification.

    Extracts multi-class subject labels and multi-hot binary tag vectors from
    data/metadata.csv and caches 16x16 center-cropped tensors in RAM.
    """

    def __init__(
        self,
        metadata_csv: str = "data/metadata.csv",
        image_dir: str = "data/images",
        target_size: int = 16,
        cache_in_ram: bool = True,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.transform = transform
        self.cache_in_ram = cache_in_ram

        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_csv}")

        self.df = pd.read_csv(metadata_csv)

        self.subjects = sorted(self.df["subject"].dropna().unique().tolist())
        self.subject2idx = {subj: i for i, subj in enumerate(self.subjects)}
        self.idx2subject = {i: subj for i, subj in enumerate(self.subjects)}
        self.num_subjects = len(self.subjects)

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

        self.base_resize = transforms.Compose([
            transforms.CenterCrop((1080, 1080)),
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.subject_labels = []
        self.tag_labels = []
        self.filenames = self.df["filename"].tolist()

        for _, row in self.df.iterrows():
            subj = row["subject"]
            self.subject_labels.append(self.subject2idx.get(subj, 0))

            tag_vec = torch.zeros(self.num_tags, dtype=torch.float32)
            if pd.notna(row["tags"]):
                for t in str(row["tags"]).split(","):
                    t_clean = t.strip()
                    if t_clean in self.tag2idx:
                        tag_vec[self.tag2idx[t_clean]] = 1.0
            self.tag_labels.append(tag_vec)

        self.subject_labels = torch.tensor(self.subject_labels, dtype=torch.long)
        self.tag_labels = torch.stack(self.tag_labels)

        self.cached_images: Optional[torch.Tensor] = None
        if self.cache_in_ram:
            self._preload_images()

    def _preload_images(self) -> None:
        """Loads and converts all images to 16x16 float tensors in memory."""
        t0 = time.time()
        tensors = []
        blank_tensor = torch.zeros((3, self.target_size, self.target_size), dtype=torch.float32)

        for fname in tqdm(self.filenames, desc="Caching 16x16 frames"):
            img_path = self.image_dir / fname
            if not img_path.exists():
                tensors.append(blank_tensor)
                continue

            try:
                with Image.open(img_path) as img:
                    tensor = self.base_resize(img.convert("RGB"))
                    tensors.append(tensor)
            except Exception:
                tensors.append(blank_tensor)

        self.cached_images = torch.stack(tensors)
        ram_mb = (self.cached_images.element_size() * self.cached_images.nelement()) / (1024 * 1024)
        elapsed = time.time() - t0
        print(f"Cached {len(self.filenames)} images in {elapsed:.2f}s ({ram_mb:.2f} MB in RAM).")

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

        return image, self.subject_labels[idx], self.tag_labels[idx]

    def save_labels_json(self, filepath: str = "checkpoints/classifier_16x16_labels.json") -> None:
        """Saves label dictionaries for inference and web app integration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "target_size": self.target_size,
            "num_subjects": self.num_subjects,
            "subjects": self.subjects,
            "subject2idx": self.subject2idx,
            "num_tags": self.num_tags,
            "tags": self.tags,
            "tag2idx": self.tag2idx,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class ResidualBlock(nn.Module):
    """Residual convolutional block designed for 16x16 feature maps."""

    def __init__(self, channels: int) -> None:
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
        return self.act2(out + residual)


class MineArt16x16Net(nn.Module):
    """Dual-Head CNN predicting subject identity and scene/entity tags."""

    def __init__(self, num_subjects: int = 101, num_tags: int = 29, base_dim: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(base_dim),
            ResidualBlock(base_dim),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(base_dim * 2),
            ResidualBlock(base_dim * 2),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 4),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(base_dim * 4),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = base_dim * 4
        self.dropout = nn.Dropout(p=0.25)

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


def compute_metrics(
    subject_logits: torch.Tensor,
    subject_targets: torch.Tensor,
    tag_logits: torch.Tensor,
    tag_targets: torch.Tensor,
    tag_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes Top-1 subject accuracy and Multi-Label Tag F1/Precision/Recall."""
    _, preds = torch.max(subject_logits, 1)
    subject_acc = (preds == subject_targets).float().mean().item()

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
    scaler: Optional[torch.amp.GradScaler] = None,
    subject_weight: float = 1.0,
    tag_weight: float = 1.0,
) -> Dict[str, float]:
    """Runs a single forward and backward pass across the training split."""
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
    avg_m = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
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
    """Evaluates validation or test loss and accuracy metrics."""
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
    avg_m = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg_m["loss"] = total_loss / n_samples
    return avg_m


def train_classifier(
    metadata_csv: str = "data/metadata.csv",
    image_dir: str = "data/images",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_dir: str = "checkpoints",
) -> None:
    """Executes the dual-head 16x16 classification training pipeline."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = MineArt16x16Dataset(
        metadata_csv=metadata_csv,
        image_dir=image_dir,
        target_size=16,
        cache_in_ram=True,
    )
    dataset.save_labels_json(str(save_path / "classifier_16x16_labels.json"))

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = MineArt16x16Net(
        num_subjects=dataset.num_subjects,
        num_tags=dataset.num_tags,
        base_dim=48,
    ).to(device)

    subject_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    tag_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_score = -1.0
    best_model_path = save_path / "best_classifier_16x16.pt"

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
        val_score = (val_res["subject_acc"] + val_res["tag_f1"]) / 2.0

        if val_score > best_val_score:
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

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_res['loss']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} | "
            f"Subj Acc: {val_res['subject_acc'] * 100:.2f}% | "
            f"Tag F1: {val_res['tag_f1'] * 100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - start_train_time
    print(f"Classification training finished in {total_time / 60:.2f} minutes.")


def predict_image(image_path: str, checkpoint_path: str = "checkpoints/best_classifier_16x16.pt") -> None:
    """Performs inference on a single test image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    subjects = checkpoint["subjects"]
    tags = checkpoint["tags"]

    model = MineArt16x16Net(num_subjects=len(subjects), num_tags=len(tags)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    with Image.open(image_path) as raw:
        tensor = preprocess(raw.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        subj_logits, tag_logits = model(tensor)
        subj_probs = torch.softmax(subj_logits, dim=-1)[0]
        top5_probs, top5_indices = torch.topk(subj_probs, 5)

        tag_probs = torch.sigmoid(tag_logits)[0]
        active_tags = [(tags[i], float(p.item())) for i, p in enumerate(tag_probs) if p >= 0.35]
        active_tags.sort(key=lambda x: x[1], reverse=True)

    print(f"Prediction for {image_path}:")
    print(f"Top Subject: {subjects[top5_indices[0]]} ({top5_probs[0] * 100:.1f}%)")
    for prob, idx in zip(top5_probs, top5_indices):
        print(f"  - {subjects[idx]}: {prob * 100:.1f}%")
    print("Tags:")
    for tag_name, score in active_tags:
        print(f"  - {tag_name}: {score * 100:.1f}%")


def main() -> None:
    """CLI entry point for classification training and inference."""
    parser = argparse.ArgumentParser(description="MineArt 16x16 Multi-Task Classifier")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--predict", type=str, default=None, help="Image path for inference")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--images", type=str, default="data/images", help="Image directory")
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


if __name__ == "__main__":
    main()
