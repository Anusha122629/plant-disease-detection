import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path

# Dataset paths
TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8

# Check dataset folders
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        "Dataset folders not found. Check dataset/train and dataset/val."
    )

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Class information
class_names = train_ds.class_names
num_classes = len(class_names)

print("Disease classes:", class_names)
print("Number of classes:", num_classes)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# Pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers for initial training
base_model.trainable = False

# Build model
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Create output directory
Path("models").mkdir(exist_ok=True)

# Save the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    "models/plant_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# Phase 1: Train the classification head
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nStarting initial training...")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Phase 2: Fine-tune the final pretrained layers
print("\nStarting fine-tuning...")

base_model.trainable = True

# Keep most pretrained layers frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stability
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Load the best saved model
best_model = tf.keras.models.load_model(
    "models/plant_disease_model.keras"
)

# Report validation performance
loss, accuracy = best_model.evaluate(val_ds, verbose=1)

print("\nTraining and fine-tuning completed!")
print(f"Best saved model validation accuracy: {accuracy * 100:.2f}%")
print("Model saved to models/plant_disease_model.keras")
print("Disease classes:", class_names)
import json

# Save disease class names in their training order
with open("models/class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

print("Disease class names saved successfully!")
