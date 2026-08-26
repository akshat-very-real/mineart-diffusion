"""
MineArt Video Frame Extractor (High-Performance Multi-Threaded Edition)

Extracts evenly-spaced frames from gameplay videos into data/images/
and tracks dataset metadata in data/metadata.csv.

Performance Optimizations:
  • Multi-threaded asynchronous PNG encoding across all CPU cores
  • Fast lossless PNG compression (IMWRITE_PNG_COMPRESSION=1)
  • Fast frame skipping via cap.grab() (skips unnecessary YUV->RGB color decode)
  • Batched metadata flushing (eliminates OS file-lock overhead)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import re
from typing import List, Optional
# pyre-ignore
import cv2

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_VIDEO_DIR = Path(r"C:\Users\Aksha\OneDrive\Videos\NVIDIA\java-runtime-epsilon")

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def clean_subject_name(name: str) -> str:
    cleaned = re.sub(r'\s*\(\d+\)$', '', name).strip()
    cleaned = re.sub(r'[_\-\s]+\d+$', '', cleaned).strip()
    return cleaned if cleaned else name


def get_target_count_for_duration(duration_seconds: float) -> int:
    """
    Determines how many frames to extract based on video duration:
      • >= 10 minutes (>= 600s)       -> 10000 frames
      • 5 to 10 minutes (300s - 600s) -> 5000 frames
      • 3 to 5 minutes (180s - 300s)  -> 1000 frames
      • 1.5 to 3 minutes (90s - 180s) -> 500 frames
      • < 1.5 minutes (< 90s)         -> 100 frames
    """
    if duration_seconds >= 600.0:
        return 10000
    elif duration_seconds >= 300.0:
        return 5000
    elif duration_seconds >= 180.0:
        return 1000
    elif duration_seconds >= 90.0:
        return 500
    else:
        return 100


def get_next_index_for_subject(output_dir: Path, subject: str) -> int:
    """
    Finds the highest existing <subject>_XXX.png number in output_dir
    and returns the next starting index (1-indexed).
    """
    pattern = re.compile(rf"^{re.escape(subject)}_(\d+)\.png$", re.IGNORECASE)
    highest = 0
    if output_dir.exists():
        for file in output_dir.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    highest = max(highest, int(match.group(1)))
    return highest + 1


def format_tags(raw_tags: Optional[List[str]]) -> str:
    """Cleans and formats tags list into a normalized comma-separated string."""
    if not raw_tags:
        return ""
    cleaned = []
    for item in raw_tags:
        for tag in str(item).split(","):
            t = tag.strip().strip("'\"")
            if t and t not in cleaned:
                cleaned.append(t)
    return ", ".join(cleaned)


def initialize_metadata(metadata_file: Path):
    """Create metadata.csv with headers if it does not exist, or ensure 'tags' column exists."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_file.exists():
        with open(metadata_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "subject",
                "source_video",
                "timestamp_seconds",
                "width",
                "height",
                "tags"
            ])
    else:
        # Check and upgrade existing header if 'tags' column is missing
        try:
            with open(metadata_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header and "tags" not in header:
                with open(metadata_file, "r", newline="", encoding="utf-8") as f:
                    content = f.read()
                lines = content.splitlines(keepends=True)
                if lines:
                    first_line = lines[0].rstrip("\r\n") + ",tags\n"
                    lines[0] = first_line
                    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
                        f.writelines(lines)
        except Exception:
            pass


def _save_image_worker(out_path: str, frame, compression_level: int = 1):
    """
    Worker function executed concurrently in background thread pool.
    Saves lossless PNG with fast zlib compression.
    """
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    metadata_file: Path,
    target_count: Optional[int] = None,
    tags: Optional[List[str]] = None,
    max_workers: int = 8,
    compression_level: int = 1
) -> int:
    """
    Extracts evenly-spaced images from a video at high speed using multi-threading,
    tagging, and fast lossless compression.
    """
    subject_name = clean_subject_name(video_path.stem.strip())
    tags_str = format_tags(tags)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        print(f"[ERROR] Could not read video properties for {video_path.name}")
        cap.release()
        return 0

    duration = total_frames / fps

    # Determine frame count (auto-duration or manual override)
    if target_count is not None and target_count > 0:
        frames_to_extract = target_count
        rule_desc = f"manual override ({target_count} frames)"
    else:
        frames_to_extract = get_target_count_for_duration(duration)
        rule_desc = f"auto-duration rule ({frames_to_extract} frames for {duration:.1f}s video)"

    num_to_extract = min(frames_to_extract, total_frames)
    start_index = get_next_index_for_subject(output_dir, subject_name)
    end_index = start_index + num_to_extract - 1

    print("=" * 100)
    print(f"[VIDEO] {video_path.name}")
    print(f" • Subject Name    : {subject_name}")
    print(f" • Tags            : {tags_str if tags_str else '(none)'}")
    print(f" • Video Duration  : {duration:.2f}s ({total_frames} frames at {fps:.1f} FPS)")
    print(f" • Extraction      : {num_to_extract} frames [{rule_desc}]")
    print(f" • Output Range    : {subject_name}_{start_index:03d}.png ... {subject_name}_{end_index:03d}.png")
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_metadata(metadata_file)

    # Calculate evenly-spaced frame indices across the full video
    if num_to_extract == 1:
        target_indices = [total_frames // 2]
    else:
        target_indices = [int(round(i * (total_frames - 1) / (num_to_extract - 1))) for i in range(num_to_extract)]

    target_set = set(target_indices)
    index_to_num = {idx: start_index + i for i, idx in enumerate(target_indices)}

    metadata_buffer: List[List] = []
    saved_count = 0
    curr_frame = 0
    progress_step = max(50, num_to_extract // 10)

    # Multi-threaded writer pool for parallel disk writes & PNG compression
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            if curr_frame not in target_set:
                # Fast skip without full color decompression
                if not cap.grab():
                    break
                curr_frame += 1
                continue

            success, frame = cap.read()
            if not success:
                break

            frame_num = index_to_num[curr_frame]
            filename = f"{subject_name}_{frame_num:03d}.png"
            out_path = output_dir / filename

            # Submit frame to background worker (frame.copy() ensures buffer safety)
            executor.submit(_save_image_worker, str(out_path), frame.copy(), compression_level)

            timestamp = curr_frame / fps
            metadata_buffer.append([
                filename,
                subject_name,
                video_path.name,
                f"{timestamp:.2f}",
                width,
                height,
                tags_str
            ])

            saved_count += 1
            if saved_count % progress_step == 0 or saved_count == num_to_extract:
                print(f"   -> Queued {saved_count}/{num_to_extract} [{filename}] at {timestamp:.1f}s")

            curr_frame += 1

    cap.release()

    # Flush all metadata in a single fast batch write
    if metadata_buffer:
        with open(metadata_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(metadata_buffer)

    print(f"[SUCCESS] Extracted and saved {saved_count} images for '{subject_name}' -> {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="MineArt - High-Speed Video Frame Extractor")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=str(DEFAULT_VIDEO_DIR), help="Directory containing recorded videos")
    parser.add_argument("--output", type=str, default=str(DEFAULT_RAW_DIR), help="Output folder for raw images (default: data/images)")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA_FILE), help="Metadata CSV path (default: data/metadata.csv)")
    parser.add_argument("--count", type=int, default=None, help="Optional manual override for number of frames to extract")
    parser.add_argument("--tags", nargs="*", default=None, help='One or more tags for the extracted images (e.g. --tags "cherry_blossom" "minecraft_block_geometry")')
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)), help="Number of concurrent image saving threads (default: CPU cores)")
    parser.add_argument("--compression", type=int, default=1, choices=range(0, 10), help="PNG compression level 0-9 (default: 1 for fastest lossless)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    metadata_file = Path(args.metadata)

    print("\n" + "=" * 60)
    print("      MineArt High-Speed Frame Extractor")
    print("=" * 60)

    # 1. Single video mode
    if args.video:
        vid_path = Path(args.video)
        if not vid_path.exists():
            print(f"[ERROR] Video file not found: {vid_path}")
            return
        extract_from_video(
            vid_path,
            output_dir,
            metadata_file,
            target_count=args.count,
            tags=args.tags,
            max_workers=args.workers,
            compression_level=args.compression
        )
        return

    # 2. Batch directory mode
    vid_dir = Path(args.video_dir)
    if not vid_dir.exists():
        print(f"[INFO] Video directory does not exist: {vid_dir}")
        print("Creating folder. Place your recorded videos inside:")
        vid_dir.mkdir(parents=True, exist_ok=True)
        print(f" -> {vid_dir}")
        return

    videos = [f for f in vid_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not videos:
        print(f"[INFO] No video files found in: {vid_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        print("Tip: Name your recording after the subject (e.g. pig.mp4, cow.mp4, ocean.mp4)")
        return

    print(f"Found {len(videos)} video file(s) in {vid_dir}:\n")
    for v in videos:
        print(f" • {v.name}")

    total = 0
    for v in videos:
        count = extract_from_video(
            v,
            output_dir,
            metadata_file,
            target_count=args.count,
            tags=args.tags,
            max_workers=args.workers,
            compression_level=args.compression
        )
        total += count

    print("\n" + "=" * 60)
    print(f"Complete: Extracted total {total} images into {output_dir}")
    print(f"Metadata recorded in {metadata_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()