import random
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Reproducible random split
random.seed(42)

# Select the extracted PlantVillage folder
root = Tk()
root.withdraw()

source_folder = filedialog.askdirectory(
    title="Select the folder containing plant disease class folders"
)

root.destroy()

if not source_folder:
    print("No folder selected. Please run the script again.")
    raise SystemExit

source = Path(source_folder)

# Destination folders inside the project
project = Path(__file__).resolve().parent
train_dir = project / "dataset" / "train"
val_dir = project / "dataset" / "val"

# Find class folders containing images
class_folders = [
    folder for folder in source.iterdir()
    if folder.is_dir()
    and any(
        image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
        for image in folder.iterdir()
    )
]

if not class_folders:
    print("No plant disease class folders found.")
    print("Select the folder directly containing the class folders.")
    raise SystemExit

total_images = 0

for class_folder in sorted(class_folders):
    images = [
        image for image in class_folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) < 2:
        print(f"Skipping {class_folder.name}: not enough images.")
        continue

    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_class_dir = train_dir / class_folder.name
    val_class_dir = val_dir / class_folder.name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(image, train_class_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_class_dir / image.name)

    total_images += len(images)

    print(
        f"{class_folder.name}: "
        f"{len(train_images)} training, "
        f"{len(val_images)} validation"
    )

print("\nDataset preparation completed!")
print(f"Total images processed: {total_images}")
print(f"Training folder: {train_dir}")
print(f"Validation folder: {val_dir}")