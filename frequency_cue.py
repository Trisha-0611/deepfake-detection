"""
frequency_cue.py
----------------
Stage 3A: Train a frequency-domain deepfake detector.

Key Insight
~~~~~~~~~~~
GAN-generated (fake) faces leave characteristic artefacts in the
frequency domain that are invisible to the naked eye but detectable
by a CNN trained on the 2D FFT log-magnitude spectrum. Real face
images have smooth, natural frequency distributions whereas deepfake
images show unnatural high-frequency spikes and periodic patterns
introduced by the generation network.

Pipeline
~~~~~~~~
  Step 1 - Precompute FFT images
            For each 224x224 face crop in dataset/faces/
            -> 2D FFT per colour channel
            -> log(1 + |F|) magnitude spectrum (visualisable as image)
            -> saved to dataset/fft_faces/real/ and dataset/fft_faces/fake/

  Step 2 - Train CNN
            MobileNetV2 backbone (pretrained ImageNet)
            -> GlobalAveragePooling -> Dense(256) -> Dropout -> Dense(1, sigmoid)
            -> saved to models/frequency_model.keras

Why MobileNetV2?
~~~~~~~~~~~~~~~~
  Intentionally different from the EfficientNetB0 used in train_spatial.py.
  Using a distinct backbone for each cue ensures the fusion module gets
  complementary (not redundant) feature representations.

Usage
~~~~~
  python frequency_cue.py                       # default settings
  python frequency_cue.py --epochs 15 --batch 32
  python frequency_cue.py --skip-precompute     # if FFT images already exist
"""

import os
import sys
import argparse
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

print(f"[INFO] TensorFlow : {tf.__version__}")
print(f"[INFO] NumPy      : {np.__version__}")


# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
FACES_DIR         = os.path.join("dataset", "faces")       # input: face crops
FFT_DIR           = os.path.join("dataset", "fft_faces")   # output: FFT images
FFT_REAL          = os.path.join(FFT_DIR, "real")
FFT_FAKE          = os.path.join(FFT_DIR, "fake")

MODELS_DIR        = "models"
MODEL_SAVE_PATH   = os.path.join(MODELS_DIR, "frequency_model.keras")
LOG_PATH          = os.path.join(MODELS_DIR, "frequency_training_log.csv")

IMAGE_SIZE        = (224, 224)
BATCH_SIZE        = 32
VAL_SPLIT         = 0.20
SEED              = 42

PHASE1_EPOCHS     = 10
PHASE1_LR         = 1e-3
PHASE2_EPOCHS     = 15
PHASE2_LR         = 1e-4
FINETUNE_LAYERS   = 20
PATIENCE          = 5
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  FFT Preprocessing
# --------------------------------------------------------------------------- #

def compute_fft_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR face crop to a 3-channel log-magnitude FFT image.

    Each colour channel is independently transformed:
      1. 2D Fast Fourier Transform
      2. Shift zero-frequency component to the centre
      3. Compute log(1 + |magnitude|) for visualisation
      4. Normalise to [0, 255] uint8

    The resulting image encodes frequency-domain information where:
      - Centre  -> low frequencies (global structure)
      - Edges   -> high frequencies (fine detail / GAN artefacts)

    Parameters
    ----------
    img_bgr : np.ndarray  – BGR image, shape (H, W, 3), dtype uint8

    Returns
    -------
    fft_img : np.ndarray  – 3-channel FFT image, shape (H, W, 3), dtype uint8
    """
    fft_channels = []

    for c in range(3):
        channel = img_bgr[:, :, c].astype(np.float32)

        # 2D FFT
        f = np.fft.fft2(channel)

        # Shift so zero-frequency is at image centre
        f_shift = np.fft.fftshift(f)

        # Log-magnitude spectrum (log compresses the dynamic range)
        magnitude = np.log1p(np.abs(f_shift))

        # Normalise to [0, 255] so we can save as uint8 image
        magnitude_norm = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        fft_channels.append(magnitude_norm)

    # Stack (H, W, 3)
    return np.stack(fft_channels, axis=-1)


def precompute_fft_dataset() -> None:
    """
    Walk through dataset/faces/real/ and dataset/faces/fake/,
    compute the FFT image for every face crop, and save it to the
    corresponding folder under dataset/fft_faces/.

    Skips files that already exist so the function is idempotent.
    """
    sources = [
        (os.path.join(FACES_DIR, "real"), FFT_REAL, "REAL"),
        (os.path.join(FACES_DIR, "fake"), FFT_FAKE, "FAKE"),
    ]

    print("\n[INFO] Pre-computing FFT images ...")
    total_saved = 0

    for src_dir, dst_dir, label in sources:
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"  [WARN] {src_dir} does not exist – skipping {label}.")
            continue

        files = [f for f in sorted(os.listdir(src_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not files:
            print(f"  [WARN] No images found in {src_dir}.")
            continue

        print(f"  {label}: {len(files)} images ...")
        saved = 0

        for fname in files:
            dst_path = os.path.join(dst_dir, fname)

            # Skip if already computed
            if os.path.exists(dst_path):
                continue

            src_path = os.path.join(src_dir, fname)
            img = cv2.imread(src_path)

            if img is None:
                continue

            # Resize to standard size (defensive – preprocess.py already does this)
            img = cv2.resize(img, IMAGE_SIZE)

            fft_img = compute_fft_image(img)
            cv2.imwrite(dst_path, fft_img)
            saved += 1

        total_saved += saved
        print(f"    Saved {saved} new FFT images to {dst_dir}")

    print(f"[INFO] FFT pre-computation complete. Total new files: {total_saved}")


# --------------------------------------------------------------------------- #
#  Dataset validation
# --------------------------------------------------------------------------- #

def check_dataset(directory: str, name: str) -> int:
    """Count images in real/ and fake/ sub-folders and exit if too few."""
    total = 0
    print(f"\n  {name} summary")
    print("  " + "-" * 32)

    for cls in ["real", "fake"]:
        cls_path = os.path.join(directory, cls)
        if not os.path.isdir(cls_path):
            n = 0
        else:
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {cls:<8}: {n} images")
        total += n

    print(f"  {'TOTAL':<8}: {total} images")
    print("  " + "-" * 32)

    if total < 10:
        print(f"\n[ERROR] Not enough images in {directory}.")
        print("        Run  python preprocess.py  first.")
        sys.exit(1)

    return total


# --------------------------------------------------------------------------- #
#  Data pipeline
# --------------------------------------------------------------------------- #

def build_datasets(batch_size: int):
    """Load FFT images into train/val tf.data.Dataset objects."""
    print("\n[INFO] Building FFT data pipeline ...")

    # MobileNetV2 expects inputs scaled to [-1, 1]
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    # Light augmentation – FFT images are symmetric, so we limit augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
    ], name="fft_augmentation")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        FFT_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        FFT_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] FFT classes : {class_names}")

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
#  Model
# --------------------------------------------------------------------------- #

def build_frequency_model():
    """
    MobileNetV2 backbone (ImageNet pretrained) with a custom binary head.

    Architecture
    ~~~~~~~~~~~~
    MobileNetV2(frozen in Phase 1)
      -> GlobalAveragePooling2D
      -> Dense(256, relu)
      -> Dropout(0.5)
      -> Dense(1, sigmoid)
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False
    print(f"[INFO] MobileNetV2 loaded  –  {len(base_model.layers)} layers, frozen.")

    inputs  = keras.Input(shape=(*IMAGE_SIZE, 3), name="fft_input")
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(256, activation="relu", name="fc1")(x)
    x       = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="FrequencyDeepfakeDetector")
    print(f"[INFO] Trainable params (Phase 1) : {sum(p.numpy().size for p in model.trainable_weights):,}")

    return model, base_model


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def _make_callbacks(log_path: str, append: bool = False) -> list:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
        CSVLogger(log_path, append=append),
    ]


def train(model, base_model, train_ds, val_ds,
          phase1_epochs: int, phase2_epochs: int, finetune_layers: int):
    """Two-phase training identical in structure to train_spatial.py."""

    # Phase 1 – head only
    print("\n" + "=" * 60)
    print("  PHASE 1 - Training head  (MobileNetV2 backbone frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc")],
    )
    model.fit(train_ds, epochs=phase1_epochs, validation_data=val_ds,
              callbacks=_make_callbacks(LOG_PATH, append=False), verbose=1)

    # Phase 2 – unfreeze top layers
    print("\n" + "=" * 60)
    print(f"  PHASE 2 - Fine-tuning top {finetune_layers} backbone layers")
    print("=" * 60)

    base_model.trainable = True
    freeze_until = len(base_model.layers) - finetune_layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= freeze_until)

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc")],
    )
    model.fit(train_ds, epochs=phase2_epochs, validation_data=val_ds,
              callbacks=_make_callbacks(LOG_PATH, append=True), verbose=1)


def evaluate(val_ds) -> None:
    """Load best checkpoint and print final validation metrics."""
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION  (best checkpoint)")
    print("=" * 60)

    if not os.path.exists(MODEL_SAVE_PATH):
        print("[WARN] Model file not found.")
        return

    best = keras.models.load_model(MODEL_SAVE_PATH)
    loss, acc, auc = best.evaluate(val_ds, verbose=0)
    print(f"  val_loss     : {loss:.4f}")
    print(f"  val_accuracy : {acc:.4f}")
    print(f"  val_auc      : {auc:.4f}")
    print("=" * 60)
    print(f"[OK] Model : {MODEL_SAVE_PATH}")
    print(f"[OK] Log   : {LOG_PATH}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION - Frequency Cue Training  (Stage 3A)")
    print("=" * 60)
    print(f"  Backbone       : MobileNetV2 (ImageNet)")
    print(f"  Input          : FFT log-magnitude spectrum")
    print(f"  Batch size     : {args.batch}")
    print(f"  Phase 1 epochs : {args.phase1_epochs}")
    print(f"  Phase 2 epochs : {args.phase2_epochs}")
    print("=" * 60)

    # 1. Validate face crops exist
    check_dataset(FACES_DIR, "Face crops")

    # 2. Pre-compute FFT images (idempotent)
    if not args.skip_precompute:
        precompute_fft_dataset()
    else:
        print("\n[INFO] Skipping FFT pre-computation (--skip-precompute set).")

    # 3. Validate FFT dataset
    check_dataset(FFT_DIR, "FFT images")

    # 4. Build data pipeline
    train_ds, val_ds, _ = build_datasets(batch_size=args.batch)

    # 5. Build model
    model, base_model = build_frequency_model()

    # 6. Train
    train(model, base_model, train_ds, val_ds,
          phase1_epochs=args.phase1_epochs,
          phase2_epochs=args.phase2_epochs,
          finetune_layers=args.finetune_layers)

    # 7. Evaluate
    evaluate(val_ds)

    print("\n[DONE] Stage 3A complete.")
    print(f"       Load with: keras.models.load_model('{MODEL_SAVE_PATH}')")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3A: Train frequency-domain deepfake detector."
    )
    parser.add_argument("--batch",           type=int, default=BATCH_SIZE)
    parser.add_argument("--phase1-epochs",   type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs",   type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--finetune-layers", type=int, default=FINETUNE_LAYERS)
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip FFT pre-computation (use if already done).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
