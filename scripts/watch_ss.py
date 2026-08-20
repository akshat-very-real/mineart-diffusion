from pathlib import Path
import shutil
import time
import csv

# CONFIGURATION
MINECRAFT_SCREENSHOTS = Path(
    r"C:\Users\Aksha\AppData\Roaming\.minecraft\screenshots"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw"

METADATA_FILE = RAW_DATA / "metadata.csv"

CHECK_INTERVAL = 1

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

# SETUP
RAW_DATA.mkdir(parents=True, exist_ok=True)

# FIND NEXT NUMBER
def get_next_number():

    highest = 0

    for file in RAW_DATA.iterdir():

        if not file.is_file():
            continue

        if not file.name.startswith("mc_"):
            continue

        try:

            number = int(file.stem.split("_")[1])

            highest = max(highest, number)

        except (IndexError, ValueError):

            continue

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
                "original_filename",
                "source_path",
                "captured_at"
            ])


def save_metadata(
    filename,
    original_filename,
    captured_at
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
            original_filename,
            str(MINECRAFT_SCREENSHOTS),
            captured_at
        ])

# GET SCREENSHOTS
def get_screenshots():

    if not MINECRAFT_SCREENSHOTS.exists():
        return []

    return [
        file
        for file in MINECRAFT_SCREENSHOTS.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

# WAIT FOR FILE TO FINISH WRITING
def wait_until_stable(
    file_path,
    checks=3,
    delay=0.5
):
    """
    Wait until the file size remains unchanged
    for several consecutive checks.

    This prevents copying a screenshot while
    Minecraft is still writing it.
    """

    previous_size = -1
    stable_count = 0

    for _ in range(20):

        try:

            current_size = file_path.stat().st_size

        except FileNotFoundError:

            return False

        if current_size == previous_size:

            stable_count += 1

        else:

            stable_count = 0

        previous_size = current_size

        if stable_count >= checks:

            return True

        time.sleep(delay)

    return False

# COPY SCREENSHOT
def copy_screenshot(source, number):

    # Make sure Minecraft has finished writing the file
    if not wait_until_stable(source):

        print(
            f"[SKIPPED] File did not become stable: "
            f"{source.name}"
        )

        return False

    new_filename = (
        f"mc_{number:06d}"
        f"{source.suffix.lower()}"
    )

    destination = RAW_DATA / new_filename

    if destination.exists():

        return False

    try:

        # Copy only after the file is stable
        shutil.copy2(
            source,
            destination
        )

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(
                source.stat().st_mtime
            )
        )

        save_metadata(
            filename=new_filename,
            original_filename=source.name,
            captured_at=timestamp
        )

        print(
            f"[ADDED] {source.name} "
            f"-> {new_filename}"
        )

        return True

    except Exception as error:

        print(
            f"[ERROR] Could not copy "
            f"{source.name}: {error}"
        )

        return False

# MAIN
def main():

    print()
    print("=" * 55)
    print("        MineArt Screenshot Collector")
    print("=" * 55)

    print()
    print("Minecraft screenshots:")
    print(MINECRAFT_SCREENSHOTS)

    print()
    print("MineArt dataset:")
    print(RAW_DATA)

    if not MINECRAFT_SCREENSHOTS.exists():

        print()
        print("[ERROR] Minecraft screenshot folder not found.")

        return

    initialize_metadata()

    next_number = get_next_number()

    processed_sources = set()

    # Ignore screenshots that existed before
    # the watcher started.
    for file in get_screenshots():

        processed_sources.add(
            str(file.resolve())
        )

    print()
    print(
        f"Starting image number: "
        f"{next_number:06d}"
    )

    print()
    print("Watcher started.")
    print("Take screenshots in Minecraft using F2.")
    print("Press Ctrl+C to stop.")
    print()

    while True:

        screenshots = get_screenshots()

        for screenshot in screenshots:

            source_id = str(
                screenshot.resolve()
            )

            if source_id in processed_sources:
                continue

            success = copy_screenshot(
                screenshot,
                next_number
            )

            if success:

                processed_sources.add(
                    source_id
                )

                next_number += 1

        time.sleep(CHECK_INTERVAL)

# ENTRY POINT
if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Watcher stopped.")