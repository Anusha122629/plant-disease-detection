import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
MODEL_PATH = Path("models/plant_disease_model.keras")
VAL_DIR = Path("dataset/val")

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Check whether required files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation dataset not found: {VAL_DIR}"
    )

# Load the trained model
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load validation images
print("Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = val_ds.class_names

# Get actual labels
y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

# Get predicted probabilities
print("Predicting plant diseases...")
predictions = model.predict(val_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 50)
print("PLANT DISEASE MODEL EVALUATION")
print("=" * 50)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)

# Create confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Plant Disease Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save evaluation results
Path("outputs").mkdir(exist_ok=True)
plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")
print("Evaluation completed!")