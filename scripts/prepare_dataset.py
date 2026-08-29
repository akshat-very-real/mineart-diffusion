"""Dataset Preprocessing Tool for MineArt Diffusion (Native 16:9 Widescreen).

Preserves the authentic 16:9 aspect ratio of the 1920x1080 Minecraft captures
without square-cropping or distortion, downscaling them to crisp high-definition
resolutions (e.g., 384x216 or 480x272) so the entire 40 GB dataset shrinks
to ~600 MB.
"""

import argparse
import os
import time
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def prepare_widescreen_dataset(
    metadata_csv: str = "data/metadata.csv",
    raw_dir: str = "data/images",
    output_dir: str = "data/processed_widescreen",
    resolution: Tuple[int, int] = (384, 224),  # Multiples of 16 for clean U-Net downsampling
    create_zip: bool = True,
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    filenames = df["filename"].tolist()

    target_w, target_h = resolution
    print(f"[*] Pre-processing {len(filenames)} images to full 16:9 widescreen ({target_w}x{target_h} px)...")

    resize_transform = transforms.Compose([
        transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    t0 = time.time()
    success_count = 0

    for fname in tqdm(filenames, desc=f"Resizing to {target_w}x{target_h}"):
        src = raw_path / fname
        dst = out_path / fname

        if dst.exists():
            success_count += 1
            continue

        if not src.exists():
            continue

        try:
            with Image.open(src) as img:
                processed = resize_transform(img.convert("RGB"))
                processed.save(dst, format="PNG", optimize=True)
                success_count += 1
        except Exception:
            continue

    elapsed = time.time() - t0
    print(f"[✓] Processed {success_count}/{len(filenames)} full 16:9 images in {elapsed:.1f}s.")

    if create_zip:
        zip_path = Path("data/mineart_widescreen_dataset.zip")
        print(f"[*] Packaging into {zip_path} for fast cloud upload...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_csv, arcname="metadata.csv")
            for img_file in tqdm(out_path.glob("*.png"), desc="Zipping widescreen dataset"):
                zf.write(img_file, arcname=f"images/{img_file.name}")

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[✓] Complete! Zip file created: {zip_path} ({zip_size_mb:.1f} MB)")
        print(f"[*] Upload {zip_path.name} ({zip_size_mb:.1f} MB) to Lightning AI / Colab!")


def main():
    parser = argparse.ArgumentParser(description="MineArt 16:9 Dataset Preparation")
    parser.add_argument("--width", type=int, default=384, help="Target width (e.g., 384 or 480)")
    parser.add_argument("--height", type=int, default=224, help="Target height (e.g., 224 or 272)")
    parser.add_argument("--no-zip", action="store_true", help="Skip creating zip archive")
    args = parser.parse_args()

    prepare_widescreen_dataset(
        resolution=(args.width, args.height),
        create_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()
