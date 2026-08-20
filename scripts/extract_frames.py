from pathlib import Path
import cv2
import csv
import re


# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your NVIDIA Minecraft recordings
VIDEO_DIR = Path(
    r"C:\Users\Aksha\OneDrive\Videos\NVIDIA\java-runtime-epsilon"
)

# Raw dataset output
RAW_DATA = PROJECT_ROOT / "data" / "raw"

# Metadata
METADATA_FILE = RAW_DATA / "metadata.csv"

# Extract 1 frame per second
FRAMES_PER_SECOND = 1

# Supported video formats
SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
}


# SETUP

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA.mkdir(parents=True, exist_ok=True)


# FIND NEXT IMAGE NUMBER

def get_next_number():
    """
    Find the highest existing mc_XXXXXX filename
    and continue from there.
    """

    highest = 0

    pattern = re.compile(r"mc_(\d+)")

    for file in RAW_DATA.iterdir():

        if not file.is_file():
            continue

        match = pattern.match(file.stem)

        if not match:
            continue

        number = int(match.group(1))

        highest = max(highest, number)

    return highest + 1


# METADATA

def initialize_metadata():

    if not METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "filename",
                "source_type",
                "source_file",
                "timestamp_seconds"
            ])


def save_metadata(
    filename,
    source_file,
    timestamp
):

    with open(
        METADATA_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            filename,
            "video",
            source_file,
            f"{timestamp:.2f}"
        ])


# EXTRACT FRAMES FROM ONE VIDEO

def extract_from_video(video_path, next_number):
    """
    Extract frames from a single video.

    Returns the next available image number.
    """

    print()
    print(f"[VIDEO] {video_path.name}")

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():

        print(
            f"[ERROR] Could not open "
            f"{video_path.name}"
        )

        return next_number

    # Video information
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:

        print(
            f"[ERROR] Could not determine FPS "
            f"for {video_path.name}"
        )

        video.release()

        return next_number

    duration = frame_count / fps

    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")

    # Number of frames to skip
    frame_interval = max(
        1,
        int(round(fps / FRAMES_PER_SECOND))
    )

    frame_number = 0
    extracted = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        # Only save selected frames
        if frame_number % frame_interval == 0:

            timestamp = frame_number / fps

            filename = (
                f"mc_{next_number:06d}.png"
            )

            output_path = RAW_DATA / filename

            # OpenCV reads frames as BGR.
            # PNG writing preserves the image correctly.
            saved = cv2.imwrite(
                str(output_path),
                frame
            )

            if saved:

                save_metadata(
                    filename=filename,
                    source_file=video_path.name,
                    timestamp=timestamp
                )

                print(
                    f"  [FRAME] "
                    f"{filename} "
                    f"({timestamp:.2f}s)"
                )

                next_number += 1
                extracted += 1

        frame_number += 1

    video.release()

    print(
        f"[DONE] Extracted "
        f"{extracted} frames"
    )

    return next_number


# MAIN

def main():

    print()
    print("=" * 60)
    print("           MineArt Video Frame Extractor")
    print("=" * 60)

    print()
    print("Video folder:")
    print(VIDEO_DIR)

    print()
    print("Output folder:")
    print(RAW_DATA)

    print()
    print(
        f"Extraction rate: "
        f"{FRAMES_PER_SECOND} frame(s) per second"
    )

    initialize_metadata()

    videos = [
        file
        for file in VIDEO_DIR.iterdir()
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    if not videos:

        print()
        print("[INFO] No videos found.")

        print()
        print(
            "Put your Minecraft recordings inside:"
        )

        print(VIDEO_DIR)

        return

    print()
    print(f"Found {len(videos)} video(s).")

    next_number = get_next_number()

    print(
        f"Starting image number: "
        f"{next_number:06d}"
    )

    for video in videos:

        next_number = extract_from_video(
            video,
            next_number
        )

    print()
    print("=" * 60)
    print("Extraction complete.")
    print("=" * 60)


# ENTRY POINT

if __name__ == "__main__":
    main()