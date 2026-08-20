"""
MineArt Diffusion - Phase 1: Dataset Creation & Preprocessing Pipeline

This script processes raw Minecraft screenshots into a clean, normalized,
reproducible ML-ready dataset split into train/validation/test folders.
It extracts metadata, detects corruptions/duplicates, center-crops/resizes
to a uniform resolution (default: 64x64), and generates statistical summaries
and visual reports.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file to identify exact duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_raw_images(raw_dir: Path) -> List[Path]:
    """Recursively search for all supported image files in raw_dir."""
    found_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found_files.append(path)
    return sorted(found_files)


def center_crop_and_resize(img: Image.Image, target_size: int) -> Image.Image:
    """
    Square-crop the image from its center to preserve aspect ratio without distortion,
    then resize to (target_size, target_size) using high-quality Lanczos resampling.
    """
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)


def process_image(
    input_path: Path,
    output_path: Path,
    target_size: int,
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """
    Validate, normalize (convert to RGB), center-crop and resize, then save as PNG.
    Returns: (is_success, error_reason, (orig_w, orig_h), orig_format)
    """
    try:
        with Image.open(input_path) as img:
            # Force verify to catch truncated / corrupted files
            img.verify()

        # Reopen after verify to perform transformations
        with Image.open(input_path) as img:
            orig_w, orig_h = img.size
            orig_format = img.format or input_path.suffix.lstrip(".").upper()

            # Convert to RGB (handles RGBA, Palette, Grayscale, etc.)
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")

            # Center-crop to square and resize
            processed_img = center_crop_and_resize(rgb_img, target_size)

            # Ensure output directory exists and save as optimized PNG
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_img.save(output_path, format="PNG", optimize=True)

            return True, None, (orig_w, orig_h), orig_format

    except Exception as e:
        return False, str(e), None, None


def split_dataset(
    items: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Dict]:
    """
    Deterministically partition valid dataset items into train, validation, and test splits.
    """
    # Normalize ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    rng = random.Random(seed)
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    n = len(shuffled_items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remaining items go to test split to guarantee total count matches exactly
    for i, item in enumerate(shuffled_items):
        if i < n_train:
            item["split"] = "train"
        elif i < n_train + n_val:
            item["split"] = "validation"
        else:
            item["split"] = "test"

    return shuffled_items


def generate_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    target_size: int,
    num_samples: int = 16,
    seed: int = 42,
) -> None:
    """Generate distribution charts and a sample contact sheet grid."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    valid_df = df[df["status"] == "valid"].copy()

    # 1. Dataset Summary Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MineArt Diffusion - Phase 1 Dataset Analysis", fontsize=16, fontweight="bold")

    # Plot A: Dataset Split Distribution
    if not valid_df.empty and "split" in valid_df.columns:
        split_counts = valid_df["split"].value_counts()
        axes[0, 0].bar(split_counts.index, split_counts.values, color=["#4CAF50", "#2196F3", "#FF9800"])
        axes[0, 0].set_title("Dataset Split Distribution")
        axes[0, 0].set_ylabel("Number of Images")
        for idx, val in enumerate(split_counts.values):
            axes[0, 0].text(idx, val + max(1, val * 0.02), str(val), ha="center", fontweight="bold")
    else:
        axes[0, 0].text(0.5, 0.5, "No valid split data", ha="center", va="center")

    # Plot B: Original Aspect Ratio Distribution
    if not valid_df.empty and "aspect_ratio" in valid_df.columns:
        axes[0, 1].hist(valid_df["aspect_ratio"], bins=20, color="#9C27B0", edgecolor="black")
        axes[0, 1].set_title("Original Aspect Ratio (Width / Height)")
        axes[0, 1].set_xlabel("Aspect Ratio")
        axes[0, 1].set_ylabel("Frequency")
    else:
        axes[0, 1].text(0.5, 0.5, "No aspect ratio data", ha="center", va="center")

    # Plot C: Original Resolution Scatter/Distribution
    if not valid_df.empty and "orig_width" in valid_df.columns:
        axes[1, 0].scatter(valid_df["orig_width"], valid_df["orig_height"], alpha=0.6, color="#009688", edgecolors="none")
        axes[1, 0].set_title("Original Resolutions (Width vs Height)")
        axes[1, 0].set_xlabel("Width (px)")
        axes[1, 0].set_ylabel("Height (px)")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, "No resolution data", ha="center", va="center")

    # Plot D: Original File Formats
    if not valid_df.empty and "format" in valid_df.columns:
        format_counts = valid_df["format"].value_counts()
        axes[1, 1].pie(format_counts.values, labels=format_counts.index, autopct="%1.1f%%", colors=["#E91E63", "#3F51B5", "#00BCD4", "#8BC34A"])
        axes[1, 1].set_title("Original File Formats")
    else:
        axes[1, 1].text(0.5, 0.5, "No format data", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plots_path = reports_dir / "dataset_summary_plots.png"
    plt.savefig(plots_path, dpi=150)
    plt.close()
    print(f"-> Saved dataset summary plots to: {plots_path}")

    # 2. Contact Sheet / Grid of Processed Images
    if not valid_df.empty:
        rng = random.Random(seed)
        sample_records = valid_df.to_dict("records")
        sample_count = min(num_samples, len(sample_records))
        selected_samples = rng.sample(sample_records, sample_count)

        grid_cols = 4
        grid_rows = (sample_count + grid_cols - 1) // grid_cols
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i in range(len(axes)):
            if i < len(selected_samples):
                sample_path = Path(selected_samples[i]["processed_path"])
                if sample_path.exists():
                    with Image.open(sample_path) as s_img:
                        axes[i].imshow(s_img)
                        axes[i].set_title(f"{selected_samples[i]['split']}\n{sample_path.name}", fontsize=8)
                        axes[i].axis("off")
                else:
                    axes[i].axis("off")
            else:
                axes[i].axis("off")

        plt.suptitle(f"MineArt Diffusion - Processed Samples ({target_size}x{target_size})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        contact_sheet_path = reports_dir / "contact_sheet.png"
        plt.savefig(contact_sheet_path, dpi=150)
        plt.close()
        print(f"-> Saved processed contact sheet to: {contact_sheet_path}")


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    generate_plots: bool = True,
) -> None:
    """Execute the full dataset preparation pipeline."""
    print("==================================================")
    print("       MineArt Diffusion - Phase 1 Pipeline       ")
    print("==================================================")
    print(f"Input Directory  : {input_dir.resolve()}")
    print(f"Output Directory : {output_dir.resolve()}")
    print(f"Target Size      : {target_size}x{target_size}")
    print(f"Split Ratios     : Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random Seed      : {seed}")
    print("--------------------------------------------------")

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    raw_files = scan_raw_images(input_dir)
    print(f"Found {len(raw_files)} candidate image file(s) in {input_dir}.")

    if len(raw_files) == 0:
        print("No image files found to process. Please copy screenshots to the input directory.")
        return

    # Tracking sets and lists
    seen_hashes = {}
    valid_records = []
    invalid_records = []
    duplicate_records = []

    print("\n[1/3] Scanning, validating and checking for duplicates...")
    for idx, file_path in enumerate(tqdm(raw_files, desc="Validating")):
        file_size = file_path.stat().st_size
        file_hash = compute_file_hash(file_path)

        # Check duplicate
        if file_hash in seen_hashes:
            orig_file = seen_hashes[file_hash]
            duplicate_records.append({
                "original_filename": file_path.name,
                "original_path": str(file_path),
                "file_size_bytes": file_size,
                "file_hash": file_hash,
                "status": "duplicate",
                "duplicate_of": str(orig_file),
                "error_reason": f"Exact duplicate of {orig_file}",
            })
            continue

        seen_hashes[file_hash] = file_path.name

        # Verify image readability and metadata
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                w, h = img.size
                fmt = img.format or file_path.suffix.lstrip(".").upper()

            valid_records.append({
                "original_filename": file_path.name,
                "original_path": str(file_path),
                "orig_width": w,
                "orig_height": h,
                "aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
                "format": fmt,
                "file_size_bytes": file_size,
                "file_hash": file_hash,
                "status": "valid",
            })
        except Exception as err:
            invalid_records.append({
                "original_filename": file_path.name,
                "original_path": str(file_path),
                "file_size_bytes": file_size,
                "file_hash": file_hash,
                "status": "corrupt",
                "error_reason": str(err),
            })

    print(f"\nScan complete:")
    print(f" - Valid unique images : {len(valid_records)}")
    print(f" - Duplicate images    : {len(duplicate_records)}")
    print(f" - Corrupt/Invalid     : {len(invalid_records)}")

    if len(valid_records) == 0:
        print("\nNo valid images to process.")
        return

    # [2/3] Assign train / validation / test splits
    print("\n[2/3] Assigning dataset splits and preprocessing images...")
    valid_records = split_dataset(
        valid_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Process and save normalized images to respective split folders
    all_processed_records = []

    # Map sequential index per split for clean filenames (e.g. train_00001.png)
    split_counters = {"train": 0, "validation": 0, "test": 0}

    for item in tqdm(valid_records, desc="Processing"):
        split_name = item["split"]
        split_counters[split_name] += 1
        new_filename = f"{split_name}_{split_counters[split_name]:05d}.png"
        out_path = output_dir / split_name / new_filename

        success, err_reason, dims, fmt = process_image(
            input_path=Path(item["original_path"]),
            output_path=out_path,
            target_size=target_size,
        )

        item["processed_filename"] = new_filename
        item["processed_path"] = str(out_path)
        item["processed_width"] = target_size
        item["processed_height"] = target_size
        item["processed_format"] = "PNG"

        all_processed_records.append(item)

    # Combine all records (valid, duplicate, invalid) for complete transparency
    total_records = all_processed_records + duplicate_records + invalid_records
    metadata_df = pd.DataFrame(total_records)

    # [3/3] Save metadata and summary statistics
    print("\n[3/3] Saving metadata and generating statistics...")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv_path = output_dir / "dataset_metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"-> Saved metadata table to: {metadata_csv_path}")

    # Build statistics JSON
    split_counts = {
        split: len([r for r in all_processed_records if r.get("split") == split])
        for split in ["train", "validation", "test"]
    }
    format_counts = metadata_df[metadata_df["status"] == "valid"]["format"].value_counts().to_dict()

    stats = {
        "summary": {
            "total_images_scanned": len(raw_files),
            "valid_processed_images": len(all_processed_records),
            "duplicate_images": len(duplicate_records),
            "corrupt_invalid_images": len(invalid_records),
        },
        "splits": split_counts,
        "parameters": {
            "target_resolution": f"{target_size}x{target_size}",
            "train_ratio": train_ratio,
            "validation_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": seed,
        },
        "original_format_distribution": format_counts,
    }

    stats_json_path = output_dir / "dataset_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"-> Saved dataset statistics to: {stats_json_path}")

    # Generate visualizations
    if generate_plots:
        generate_visualizations(
            df=metadata_df,
            output_dir=output_dir,
            target_size=target_size,
            num_samples=16,
            seed=seed,
        )

    print("\n==================================================")
    print("           Phase 1 Pipeline Completed!            ")
    print(f" Processed images ready at: {output_dir.resolve()}")
    print("==================================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MineArt Diffusion - Phase 1: Dataset Creation & Preprocessing"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Path to folder containing raw screenshots (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Path to folder where processed dataset will be stored (default: data/processed)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Target square resolution width/height in pixels (default: 64)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of valid images for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of valid images for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of valid images for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset split (default: 42)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic generation of summary plots and contact sheet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_size=args.size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )
