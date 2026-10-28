import json
from pathlib import Path

# Find the training dataset folders
train_dir = Path("dataset/train")
models_dir = Path("models")

if not train_dir.exists():
    raise FileNotFoundError("Could not find dataset/train")

# TensorFlow loads class folders in alphabetical order
class_names = sorted(
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
)

# Save the class names
models_dir.mkdir(exist_ok=True)

output_file = models_dir / "class_names.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("Class names saved successfully!")
print(f"Number of classes: {len(class_names)}")
print(f"Saved to: {output_file.resolve()}")
print("\nClass names:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")
