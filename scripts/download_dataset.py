import kagglehub
import shutil
from pathlib import Path

# Your MineArt project
PROJECT_ROOT = Path(r"D:\mineart-diffusion")

# Destination for the dataset
DESTINATION = PROJECT_ROOT / "data" / "raw"

# Create destination if it doesn't exist
DESTINATION.mkdir(parents=True, exist_ok=True)

print("Downloading Minecraft dataset...")

# Download latest version from Kaggle
path = kagglehub.dataset_download(
    "sqdartemy/minecraft-screenshots-dataset-with-features"
)

source = Path(path)

print(f"Downloaded to: {source}")
print(f"Copying dataset to: {DESTINATION}")

# Copy everything from Kaggle's downloaded dataset
for item in source.iterdir():
    destination = DESTINATION / item.name

    if item.is_dir():
        shutil.copytree(item, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(item, destination)

print("\nDataset successfully placed in:")
print(DESTINATION)