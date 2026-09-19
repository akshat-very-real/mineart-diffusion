"""MineArt High-Efficiency Multi-Crop Preprocessing Engine.

Extracts Left, Center, and Right 1080x1080 scene windows from 1920x1080 Minecraft captures
and downscales them to crisp 64x64 pixel-art training images.
Expands ~23,200 captures to ~69,600 training images without discarding widescreen context.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm


def process_single_image(args_tuple: Tuple[str, str, str]) -> Optional[List[Dict]]:
    filename, raw_dir, out_dir = args_tuple
    src_path = Path(raw_dir) / filename
    if not src_path.exists():
        return None

    stem = Path(filename).stem
    ext = Path(filename).suffix or ".png"

    out_left = f"{stem}_L{ext}"
    out_center = f"{stem}_C{ext}"
    out_right = f"{stem}_R{ext}"

    p_out = Path(out_dir)
    target_l = p_out / out_left
    target_c = p_out / out_center
    target_r = p_out / out_right

    # If all three exist, skip computation
    if target_l.exists() and target_c.exists() and target_r.exists():
        return [
            {"filename": out_left, "crop_type": "left"},
            {"filename": out_center, "crop_type": "center"},
            {"filename": out_right, "crop_type": "right"},
        ]

    try:
        with Image.open(src_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size

            if w < h:
                # Square or portrait fallback
                square = rgb.resize((64, 64), Image.Resampling.BILINEAR)
                square.save(target_c, format="PNG", optimize=True)
                return [{"filename": out_center, "crop_type": "center"}]

            # 3 scene crops (Left, Center, Right)
            # Left: (0, 0, h, h)
            # Center: ((w - h)//2, 0, (w + h)//2, h)
            # Right: (w - h, 0, w, h)
            crop_l = rgb.crop((0, 0, h, h)).resize((64, 64), Image.Resampling.BILINEAR)
            crop_c = rgb.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((64, 64), Image.Resampling.BILINEAR)
            crop_r = rgb.crop((w - h, 0, w, h)).resize((64, 64), Image.Resampling.BILINEAR)

            crop_l.save(target_l, format="PNG", optimize=True)
            crop_c.save(target_c, format="PNG", optimize=True)
            crop_r.save(target_r, format="PNG", optimize=True)

            return [
                {"filename": out_left, "crop_type": "left"},
                {"filename": out_center, "crop_type": "center"},
                {"filename": out_right, "crop_type": "right"},
            ]
    except Exception:
        return None


def run_multicrop_pipeline(
    metadata_csv: str = "data/metadata.csv",
    raw_dir: str = "data/images",
    output_dir: str = "data/processed_64x64",
    output_metadata: str = "data/metadata_multicrop.csv",
    max_workers: int = 8,
    limit: Optional[int] = None,
):
    print("=" * 65)
    print("      MINEART MULTI-CROP DATASET EXPANSION PIPELINE")
    print("=" * 65)
    print(f" Source metadata:  {metadata_csv}")
    print(f" Raw image dir:    {raw_dir}")
    print(f" Target directory: {output_dir}")
    print(f" CPU Workers:      {max_workers}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()

    total_sources = len(df)
    print(f" Processing {total_sources} captures -> target ~{total_sources * 3} crops (64x64 px)...")

    work_items = [(row["filename"], raw_dir, output_dir) for _, row in df.iterrows()]

    t0 = time.time()
    new_rows = []

    # Map original metadata row by filename
    metadata_dict = df.set_index("filename").to_dict(orient="index")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, work_items, chunksize=64),
            total=len(work_items),
            desc="Multi-Cropping 1080p -> 64x64",
        ))

    success_sources = 0
    for idx, res in enumerate(results):
        if res is not None:
            success_sources += 1
            orig_filename = work_items[idx][0]
            orig_meta = metadata_dict.get(orig_filename, {})
            for item in res:
                row_copy = dict(orig_meta)
                row_copy["filename"] = item["filename"]
                row_copy["crop_type"] = item["crop_type"]
                row_copy["width"] = 64
                row_copy["height"] = 64
                new_rows.append(row_copy)

    elapsed = time.time() - t0
    expanded_df = pd.DataFrame(new_rows)
    expanded_df.to_csv(output_metadata, index=False)

    print("\n" + "=" * 65)
    print("              MULTI-CROP EXPANSION COMPLETE")
    print("=" * 65)
    print(f"  Source captures processed: {success_sources} / {total_sources}")
    print(f"  Total 64x64 images created: {len(expanded_df):,}")
    print(f"  New metadata saved to:      {output_metadata}")
    print(f"  Elapsed time:               {elapsed:.1f}s ({success_sources / elapsed:.1f} captures/sec)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="MineArt Multi-Crop Preprocessing")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv", help="Source metadata CSV")
    parser.add_argument("--raw-dir", type=str, default="data/images", help="Raw images directory")
    parser.add_argument("--output-dir", type=str, default="data/processed_64x64", help="Processed 64x64 output dir")
    parser.add_argument("--output-metadata", type=str, default="data/metadata_multicrop.csv", help="Expanded metadata output CSV")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    run_multicrop_pipeline(
        metadata_csv=args.metadata,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_metadata=args.output_metadata,
        max_workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
