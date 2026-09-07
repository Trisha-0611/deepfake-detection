"""
train_spatial.py
----------------
Stage 2: Train a spatial deepfake detector using transfer learning.

Architecture
~~~~~~~~~~~~
  EfficientNetB0 (ImageNet weights, frozen in Phase 1)
      -> GlobalAveragePooling2D
      -> Dense(256, relu) + Dropout(0.5)
      -> Dense(1, sigmoid)          <- REAL=0, FAKE=1

Training strategy (2-phase)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Phase 1  :  Freeze backbone entirely.  Train the custom head only.
              Fast convergence, avoids corrupting pretrained weights.
  Phase 2  :  Unfreeze the top N layers of the backbone.  Fine-tune
              with a very small learning rate.

Input
~~~~~
  dataset/faces/real/   <- 224x224 face crops labelled REAL (0)
  dataset/faces/fake/   <- 224x224 face crops labelled FAKE (1)

Output
~~~~~~
  models/spatial_model.keras   <- best checkpoint saved automatically
  models/training_log.txt      <- per-epoch metrics

Usage
~~~~~
  python train_spatial.py
  python train_spatial.py --epochs 20 --batch 16 --finetune-layers 30
"""

import os
import sys
import argparse
import datetime

# ------------------------------------------------------------------ #
# TensorFlow / Keras imports
# ------------------------------------------------------------------ #
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress verbose TF logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

print(f"[INFO] TensorFlow version : {tf.__version__}")
print(f"[INFO] Keras version      : {keras.__version__}")


# =========================================================================== #
#  CONFIGURATION  –  change these values to tune training
# =========================================================================== #
FACES_DIR         = os.path.join("dataset", "faces")   # parent of real/ & fake/
MODELS_DIR        = "models"
MODEL_SAVE_PATH   = os.path.join(MODELS_DIR, "spatial_model.keras")
LOG_PATH          = os.path.join(MODELS_DIR, "training_log.csv")

IMAGE_SIZE        = (224, 224)       # must match preprocess.py output
BATCH_SIZE        = 32               # reduce to 16 if you run out of GPU/RAM
VAL_SPLIT         = 0.20            # 20 % of data used for validation
SEED              = 42

# Phase 1 – head-only training
PHASE1_EPOCHS     = 10
PHASE1_LR         = 1e-3

# Phase 2 – backbone fine-tuning
PHASE2_EPOCHS     = 20
PHASE2_LR         = 1e-4
FINETUNE_LAYERS   = 20              # how many top EfficientNetB0 layers to unfreeze

# Early-stopping patience (stops training if val_loss doesn't improve)
PATIENCE          = 5
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def count_images(directory: str) -> dict:
    """Count images in each sub-folder of `directory`."""
    counts = {}
    if not os.path.isdir(directory):
        return counts
    for cls in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            counts[cls] = n
    return counts


def check_dataset() -> None:
    """
    Verify that the faces directory exists and has enough images to train.
    Exits with a helpful message if the dataset is missing or too small.
    """
    counts = count_images(FACES_DIR)

    if not counts:
        print(f"\n[ERROR] No class folders found in: {FACES_DIR}")
        print("        Run  python preprocess.py  first to generate face crops.")
        sys.exit(1)

    print("\n  Dataset summary")
    print("  " + "-" * 30)
    total = 0
    for cls, n in counts.items():
        print(f"  {cls:<10} : {n} images")
        total += n
    print(f"  {'TOTAL':<10} : {total} images")
    print("  " + "-" * 30)

    if total < 10:
        print("\n[ERROR] Not enough images to train (need at least 10).")
        print("        Make sure your FF++ videos are in dataset/raw/ and run")
        print("        python preprocess.py  to extract faces.")
        sys.exit(1)

    if len(counts) < 2:
        print("\n[ERROR] Need at least 2 classes (real and fake) to train.")
        sys.exit(1)


# --------------------------------------------------------------------------- #
#  Data pipeline
# --------------------------------------------------------------------------- #

def build_datasets(batch_size: int):
    """
    Load train and validation datasets from the faces directory.

    Uses `image_dataset_from_directory` which automatically:
      - assigns labels based on sub-folder names (real=0, fake=1 sorted alpha)
      - applies the val_split
      - returns tf.data.Dataset objects ready for training

    Returns
    -------
    train_ds, val_ds, class_names
    """
    print("\n[INFO] Building data pipeline ...")

    # Augmentation layers – applied ONLY during training
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),           # +/- 5% rotation
        layers.RandomBrightness(0.1),          # +/- 10% brightness
        layers.RandomContrast(0.1),
    ], name="augmentation")

    # Preprocessing function expected by EfficientNetB0
    # (scales pixels to [-1, 1] using the ImageNet normalisation)
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    # Load training subset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        FACES_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",          # returns float32 labels: 0.0 or 1.0
        shuffle=True,
    )

    # Load validation subset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        FACES_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes: {class_names}  (label order: alphabetical)")

    # Apply augmentation to training set only, then preprocess both
    train_ds = (
        train_ds
        .map(lambda x, y: (data_augmentation(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names


# --------------------------------------------------------------------------- #
#  Model builder
# --------------------------------------------------------------------------- #

def build_model() -> keras.Model:
    """
    Build and return the spatial deepfake detection model.

    Architecture:
      EfficientNetB0 (pretrained, backbone)
        -> GlobalAveragePooling2D
        -> Dense(256, relu)
        -> Dropout(0.5)
        -> Dense(1, sigmoid)   <- probability of FAKE
    """
    # Load the EfficientNetB0 backbone pretrained on ImageNet.
    # include_top=False means we remove the original 1000-class head.
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )

    # Freeze entire backbone for Phase 1
    base_model.trainable = False
    print(f"[INFO] EfficientNetB0 loaded  –  {len(base_model.layers)} layers, frozen.")

    # Build our custom classification head on top
    inputs  = keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x       = base_model(inputs, training=False)   # keep BN in inference mode
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(256, activation="relu", name="fc1")(x)
    x       = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="SpatialDeepfakeDetector")

    total_params   = model.count_params()
    trainable      = sum(p.numpy().size for p in model.trainable_weights)
    non_trainable  = total_params - trainable
    print(f"[INFO] Total params      : {total_params:,}")
    print(f"[INFO] Trainable params  : {trainable:,}  (head only in Phase 1)")
    print(f"[INFO] Non-trainable     : {non_trainable:,}")

    return model, base_model


# --------------------------------------------------------------------------- #
#  Training phases
# --------------------------------------------------------------------------- #

def phase1_train(model, base_model, train_ds, val_ds, epochs: int, lr: float):
    """
    Phase 1: train the head only with the backbone frozen.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 – Training head  (backbone frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = _make_callbacks(phase=1)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def phase2_finetune(model, base_model, train_ds, val_ds,
                    epochs: int, lr: float, finetune_layers: int):
    """
    Phase 2: unfreeze the top `finetune_layers` of the backbone and
    fine-tune everything end-to-end with a very small learning rate.
    """
    print("\n" + "=" * 60)
    print(f"  PHASE 2 – Fine-tuning top {finetune_layers} backbone layers")
    print("=" * 60)

    # Unfreeze the backbone partially
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = total_layers - finetune_layers

    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= freeze_until)

    trainable_now = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Unfrozen {trainable_now}/{total_layers} backbone layers.")

    # Recompile with a much lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = _make_callbacks(phase=2)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def _make_callbacks(phase: int) -> list:
    """Build the Keras callbacks used in both training phases."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Stop early if val_loss stops improving
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR if val_loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # Append per-epoch metrics to a CSV log file
        CSVLogger(LOG_PATH, append=(phase == 2)),
    ]
    return callbacks


# --------------------------------------------------------------------------- #
#  Final evaluation
# --------------------------------------------------------------------------- #

def evaluate(model_path: str, val_ds) -> None:
    """Load the best saved checkpoint and print final metrics."""
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION  (best saved checkpoint)")
    print("=" * 60)

    if not os.path.exists(model_path):
        print("[WARN] Model file not found – skipping evaluation.")
        return

    best_model = keras.models.load_model(model_path)
    results = best_model.evaluate(val_ds, verbose=0)

    metric_names = ["loss", "accuracy", "auc"]
    for name, value in zip(metric_names, results):
        print(f"  val_{name:<12} : {value:.4f}")

    print("=" * 60)
    print(f"[OK] Best model saved at: {model_path}")
    print(f"[OK] Training log saved at: {LOG_PATH}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION - Spatial CNN Training  (Stage 2)")
    print("=" * 60)
    print(f"  Backbone         : EfficientNetB0 (ImageNet)")
    print(f"  Image size       : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(f"  Batch size       : {args.batch}")
    print(f"  Phase 1 epochs   : {args.phase1_epochs}")
    print(f"  Phase 2 epochs   : {args.phase2_epochs}")
    print(f"  Fine-tune layers : {args.finetune_layers}")
    print(f"  Model output     : {MODEL_SAVE_PATH}")
    print("=" * 60)

    # 1. Validate dataset
    check_dataset()

    # 2. Build tf.data pipelines
    train_ds, val_ds, class_names = build_datasets(batch_size=args.batch)

    # 3. Build model
    model, base_model = build_model()

    # 4. Phase 1 – head-only training
    phase1_train(
        model, base_model, train_ds, val_ds,
        epochs=args.phase1_epochs,
        lr=PHASE1_LR,
    )

    # 5. Phase 2 – fine-tuning
    phase2_finetune(
        model, base_model, train_ds, val_ds,
        epochs=args.phase2_epochs,
        lr=PHASE2_LR,
        finetune_layers=args.finetune_layers,
    )

    # 6. Evaluate best checkpoint
    evaluate(MODEL_SAVE_PATH, val_ds)

    print("\n[DONE] Stage 2 complete.")
    print(f"       Load the model with:")
    print(f"       model = tf.keras.models.load_model('{MODEL_SAVE_PATH}')")


# --------------------------------------------------------------------------- #
#  Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Train spatial deepfake detector (EfficientNetB0)."
    )
    parser.add_argument("--batch",           type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--phase1-epochs",   type=int, default=PHASE1_EPOCHS,
                        help=f"Phase 1 epochs – head only (default: {PHASE1_EPOCHS})")
    parser.add_argument("--phase2-epochs",   type=int, default=PHASE2_EPOCHS,
                        help=f"Phase 2 epochs – fine-tune (default: {PHASE2_EPOCHS})")
    parser.add_argument("--finetune-layers", type=int, default=FINETUNE_LAYERS,
                        help=f"Top N backbone layers to unfreeze in Phase 2 (default: {FINETUNE_LAYERS})")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = parse_args()
    main(args)
