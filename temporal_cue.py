"""
temporal_cue.py
---------------
Stage 3B: Train a temporal deepfake detector using BiLSTM.

Key Insight
~~~~~~~~~~~
Deepfake videos often show unnatural temporal patterns across frames:
  - Inconsistent blinking and eye movement
  - Head-pose jitter between consecutive frames
  - Subtle per-frame lighting or texture flicker from the generation network
  - Temporal boundary artefacts at video clip edges

A single-frame classifier (spatial or frequency) cannot detect these patterns.
A Bidirectional LSTM over per-frame spatial features can.

Pipeline
~~~~~~~~
  Step 1 - Feature extraction
            Use EfficientNetB0 backbone as a frozen feature extractor.
            Every face crop -> 1280-dimensional feature vector.
            Features saved to dataset/temporal_features/ as .npy files.

  Step 2 - Sequence building
            Face crop filenames encode the source video and frame number:
              {video_name}_frame{N:06d}_face{M:02d}.jpg
            Group crops by video_name, sort by frame number,
            take one crop per frame (the first face detected).
            Pad / truncate sequences to SEQ_LEN steps.

  Step 3 - Train BiLSTM
            Input  : (batch, SEQ_LEN, 1280) feature sequences
            Model  : Masking -> BiLSTM(128) -> Dense(64) -> Dropout -> Dense(1)
            Output : P(fake) per video clip
            Saved  : models/temporal_model.keras

Usage
~~~~~
  python temporal_cue.py                             # default settings
  python temporal_cue.py --seq-len 15 --epochs 25
  python temporal_cue.py --skip-extraction           # if features already saved
"""

import os
import sys
import argparse
import re
from collections import defaultdict

import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

print(f"[INFO] TensorFlow : {tf.__version__}")
print(f"[INFO] NumPy      : {np.__version__}")


# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
FACES_DIR       = os.path.join("dataset", "faces")
FEATURES_DIR    = os.path.join("dataset", "temporal_features")

MODELS_DIR      = "models"
SPATIAL_MODEL   = os.path.join(MODELS_DIR, "spatial_model.keras")   # preferred
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "temporal_model.keras")
LOG_PATH        = os.path.join(MODELS_DIR, "temporal_training_log.csv")

IMAGE_SIZE      = (224, 224)
FEATURE_DIM     = 1280          # EfficientNetB0 output after GlobalAveragePooling
SEQ_LEN         = 10            # frames per clip used for training
BATCH_SIZE      = 16            # smaller batch: sequences are heavier than images
VAL_SPLIT       = 0.20
SEED            = 42

EPOCHS          = 25
LR              = 1e-3
PATIENCE        = 7
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  Filename parsing
# --------------------------------------------------------------------------- #

# Filename format: {video_name}_frame{N:06d}_face{M:02d}.jpg
_FRAME_RE = re.compile(r"^(.+)_frame(\d+)_face(\d+)\.[a-zA-Z]+$")


def parse_filename(filename: str):
    """
    Parse a face-crop filename into its components.

    Returns (video_name, frame_idx, face_idx) or None if the filename
    does not match the expected format.
    """
    m = _FRAME_RE.match(filename)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


# --------------------------------------------------------------------------- #
#  Backbone feature extractor
# --------------------------------------------------------------------------- #

def load_feature_extractor() -> keras.Model:
    """
    Return an EfficientNetB0-based feature extractor (frozen).

    Strategy:
      1. Try to load the trained spatial model from Stage 2 and reuse
         its EfficientNetB0 backbone.  This gives us a domain-adapted
         feature extractor that already understands face artefacts.
      2. If the spatial model doesn't exist yet, fall back to a fresh
         EfficientNetB0 with ImageNet weights.  Still useful because
         ImageNet features generalise well to face content.
    """
    if os.path.exists(SPATIAL_MODEL):
        print(f"[INFO] Loading trained spatial model from: {SPATIAL_MODEL}")
        spatial = keras.models.load_model(SPATIAL_MODEL)

        # Extract just the EfficientNetB0 backbone sub-model
        backbone = None
        for layer in spatial.layers:
            if "efficientnet" in layer.name.lower():
                backbone = layer
                break

        if backbone is not None:
            # Build extractor: backbone -> GAP -> feature vector
            inp = keras.Input(shape=(*IMAGE_SIZE, 3), name="temporal_input")
            x   = backbone(inp, training=False)
            x   = layers.GlobalAveragePooling2D(name="gap")(x)
            extractor = keras.Model(inp, x, name="FeatureExtractor")
            backbone.trainable = False
            print(f"[INFO] Using trained backbone from spatial model.")
            return extractor

    # Fallback: fresh EfficientNetB0
    print("[INFO] Spatial model not found – using fresh EfficientNetB0 (ImageNet).")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",          # includes GAP – output shape: (batch, 1280)
    )
    base.trainable = False
    return base


def preprocess_for_efficientnet(img_bgr: np.ndarray) -> np.ndarray:
    """Resize and normalise a BGR image for EfficientNetB0."""
    img = cv2.resize(img_bgr, IMAGE_SIZE)
    img = img[:, :, ::-1].astype(np.float32)   # BGR -> RGB
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


# --------------------------------------------------------------------------- #
#  Step 1: Extract and cache per-frame features
# --------------------------------------------------------------------------- #

def extract_features(extractor: keras.Model) -> None:
    """
    Extract EfficientNetB0 feature vectors for every face crop and save
    them as .npy files under dataset/temporal_features/real/ and .../fake/.

    The .npy filename mirrors the source .jpg filename for easy matching.
    Already-extracted files are skipped (idempotent).
    """
    sources = [
        (os.path.join(FACES_DIR, "real"),
         os.path.join(FEATURES_DIR, "real"), "REAL"),
        (os.path.join(FACES_DIR, "fake"),
         os.path.join(FEATURES_DIR, "fake"), "FAKE"),
    ]

    print("\n[INFO] Extracting temporal features ...")

    for src_dir, dst_dir, label in sources:
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"  [WARN] {src_dir} not found – skipping {label}.")
            continue

        files = [f for f in sorted(os.listdir(src_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not files:
            print(f"  [WARN] No images in {src_dir}.")
            continue

        pending = [f for f in files
                   if not os.path.exists(
                       os.path.join(dst_dir,
                                    os.path.splitext(f)[0] + ".npy"))]

        print(f"  {label}: {len(files)} crops, {len(pending)} to extract ...")

        # Process in mini-batches for speed
        mini_batch = 32
        for start in range(0, len(pending), mini_batch):
            batch_files = pending[start : start + mini_batch]
            imgs = []

            for fname in batch_files:
                img = cv2.imread(os.path.join(src_dir, fname))
                if img is None:
                    imgs.append(np.zeros((*IMAGE_SIZE, 3), dtype=np.float32))
                    continue
                imgs.append(preprocess_for_efficientnet(img))

            # Stack into batch tensor and run forward pass
            batch_tensor = np.stack(imgs, axis=0)
            features = extractor.predict(batch_tensor, verbose=0)  # (B, 1280)

            # Save each feature vector
            for fname, feat in zip(batch_files, features):
                npy_name = os.path.splitext(fname)[0] + ".npy"
                np.save(os.path.join(dst_dir, npy_name), feat)

        print(f"    Done -> {dst_dir}")

    print("[INFO] Feature extraction complete.")


# --------------------------------------------------------------------------- #
#  Step 2: Build sequence dataset
# --------------------------------------------------------------------------- #

def load_video_sequences(seq_len: int):
    """
    Group extracted features by video name and build fixed-length sequences.

    Two modes
    ---------
    1. **Video mode** (FF++ filenames): filenames follow the pattern
       ``{video_name}_frame{N:06d}_face{M:02d}.npy``.
       Frames are grouped by video, sorted by frame index.

    2. **Image mode** (demo / still-image datasets): filenames do NOT match
       the video pattern (e.g. ``real_0000.npy``).
       Consecutive features are grouped into sliding windows of *seq_len*
       so the BiLSTM still sees multi-step input sequences.

    Returns
    -------
    X : np.ndarray  shape (N_clips, seq_len, FEATURE_DIM)
    y : np.ndarray  shape (N_clips,)   – 0=real, 1=fake
    """
    categories = [("real", 0), ("fake", 1)]
    all_X, all_y = [], []

    for cls_name, label in categories:
        feat_dir = os.path.join(FEATURES_DIR, cls_name)

        if not os.path.isdir(feat_dir):
            print(f"  [WARN] Feature dir not found: {feat_dir}")
            continue

        npy_files = [f for f in sorted(os.listdir(feat_dir))
                     if f.endswith(".npy")]

        if not npy_files:
            continue

        # ---- Try video-mode grouping first -------------------------------- #
        video_dict = defaultdict(list)   # video_name -> [(frame_idx, path), ...]

        for fname in npy_files:
            stem = os.path.splitext(fname)[0] + ".jpg"
            parsed = parse_filename(stem)
            if parsed is None:
                continue
            video_name, frame_idx, face_idx = parsed
            if face_idx == 0:
                video_dict[video_name].append(
                    (frame_idx, os.path.join(feat_dir, fname))
                )

        if video_dict:
            # ---- Video mode: FF++ style ----------------------------------- #
            print(f"  {cls_name.upper()}: {len(video_dict)} video(s) found.")

            for video_name, frame_list in video_dict.items():
                frame_list.sort(key=lambda t: t[0])
                feat_vectors = [np.load(p) for _, p in frame_list]

                if not feat_vectors:
                    continue

                if len(feat_vectors) >= seq_len:
                    seq = np.stack(feat_vectors[:seq_len], axis=0)
                else:
                    pad = seq_len - len(feat_vectors)
                    seq = np.stack(feat_vectors + [np.zeros(FEATURE_DIM)] * pad,
                                   axis=0)

                all_X.append(seq)
                all_y.append(label)

        else:
            # ---- Image mode: demo / still-image fallback ----------------- #
            print(f"  {cls_name.upper()}: {len(npy_files)} individual feature(s) found "
                  f"(image-mode – grouping into sliding-window sequences).")

            # Load all feature vectors for this class
            feats = []
            for fname in npy_files:
                vec = np.load(os.path.join(feat_dir, fname))
                feats.append(vec)

            if len(feats) == 0:
                continue

            if len(feats) < seq_len:
                # Too few images: repeat the whole list until we have seq_len
                while len(feats) < seq_len:
                    feats = feats + feats
                feats = feats[:seq_len]

            # Sliding-window: step = seq_len // 2  (50 % overlap)
            step = max(1, seq_len // 2)
            for start in range(0, len(feats) - seq_len + 1, step):
                window = feats[start : start + seq_len]
                all_X.append(np.stack(window, axis=0))
                all_y.append(label)

            # Always add at least one sequence (tail of the list)
            if not all_X or (all_y and all_y[-1] != label):
                tail = feats[-seq_len:]
                if len(tail) < seq_len:
                    tail += [np.zeros(FEATURE_DIM)] * (seq_len - len(tail))
                all_X.append(np.stack(tail, axis=0))
                all_y.append(label)

    if not all_X:
        print("\n[ERROR] No sequences found. Make sure you have face crops and")
        print("        have run feature extraction.")
        sys.exit(1)

    X = np.stack(all_X, axis=0).astype(np.float32)   # (N, seq_len, 1280)
    y = np.array(all_y, dtype=np.float32)             # (N,)

    print(f"\n  Sequence dataset: {X.shape[0]} clips, shape {X.shape}")
    return X, y



def split_dataset(X: np.ndarray, y: np.ndarray, val_split: float, seed: int):
    """Shuffle and split X, y into train/val arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    val_n  = max(1, int(len(X) * val_split))
    X_val, y_val     = X[:val_n], y[:val_n]
    X_train, y_train = X[val_n:], y[val_n:]

    print(f"  Train : {len(X_train)} clips")
    print(f"  Val   : {len(X_val)} clips")
    return X_train, y_train, X_val, y_val


# --------------------------------------------------------------------------- #
#  Step 3: Build temporal model
# --------------------------------------------------------------------------- #

def build_temporal_model(seq_len: int) -> keras.Model:
    """
    Bidirectional LSTM classifier on top of pre-extracted feature sequences.

    Architecture
    ~~~~~~~~~~~~
    Input  : (seq_len, FEATURE_DIM)
    Masking: ignores zero-padded time steps
    BiLSTM(128) -> Dense(64, relu) -> Dropout(0.5) -> Dense(1, sigmoid)

    The Masking layer is important: padded frames (all zeros) are ignored
    during the LSTM computation, so the model learns from actual frame count.
    """
    inputs = keras.Input(shape=(seq_len, FEATURE_DIM), name="feature_sequence")

    # Masking: time steps that are all zeros (padding) are skipped
    x = layers.Masking(mask_value=0.0, name="masking")(inputs)

    # Bidirectional LSTM – captures forward and backward temporal context
    x = layers.Bidirectional(
        layers.LSTM(128, name="lstm"),
        name="bilstm"
    )(x)

    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="TemporalDeepfakeDetector")

    total = model.count_params()
    print(f"[INFO] Temporal model built  –  {total:,} trainable params")

    return model


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train_temporal(model: keras.Model,
                   X_train, y_train, X_val, y_val,
                   epochs: int) -> None:
    """Compile and train the temporal BiLSTM model."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
        CSVLogger(LOG_PATH, append=False),
    ]

    print("\n" + "=" * 60)
    print("  Training BiLSTM temporal model ...")
    print("=" * 60)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )


def evaluate_temporal(val_X, val_y) -> None:
    """Load best checkpoint and evaluate."""
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION  (best checkpoint)")
    print("=" * 60)

    if not os.path.exists(MODEL_SAVE_PATH):
        print("[WARN] Model file not found.")
        return

    best = keras.models.load_model(MODEL_SAVE_PATH)
    loss, acc, auc = best.evaluate(val_X, val_y, verbose=0)
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
    print("  DEEPFAKE DETECTION - Temporal Cue Training  (Stage 3B)")
    print("=" * 60)
    print(f"  Feature extractor : EfficientNetB0 (frozen)")
    print(f"  Sequence length   : {args.seq_len} frames per clip")
    print(f"  LSTM units        : 128 (Bidirectional)")
    print(f"  Batch size        : {args.batch}")
    print(f"  Epochs            : {args.epochs}")
    print("=" * 60)

    # 1. Check face crops exist
    for cls in ["real", "fake"]:
        p = os.path.join(FACES_DIR, cls)
        if not os.path.isdir(p) or not os.listdir(p):
            print(f"\n[ERROR] No face crops found in {p}.")
            print("        Run  python preprocess.py  first.")
            sys.exit(1)

    # 2. Load feature extractor
    extractor = load_feature_extractor()

    # 3. Extract per-frame features (idempotent)
    if not args.skip_extraction:
        extract_features(extractor)
    else:
        print("\n[INFO] Skipping feature extraction (--skip-extraction set).")

    # 4. Build sequence dataset
    print("\n[INFO] Building sequence dataset ...")
    X, y = load_video_sequences(seq_len=args.seq_len)

    X_train, y_train, X_val, y_val = split_dataset(X, y, VAL_SPLIT, SEED)

    # 5. Build temporal model
    model = build_temporal_model(seq_len=args.seq_len)
    model.summary()

    # 6. Train
    train_temporal(model, X_train, y_train, X_val, y_val, epochs=args.epochs)

    # 7. Evaluate
    evaluate_temporal(X_val, y_val)

    print("\n[DONE] Stage 3B complete.")
    print(f"       Load with: keras.models.load_model('{MODEL_SAVE_PATH}')")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3B: Train temporal BiLSTM deepfake detector."
    )
    parser.add_argument("--seq-len",         type=int, default=SEQ_LEN,
                        help=f"Frames per clip (default: {SEQ_LEN})")
    parser.add_argument("--batch",           type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--epochs",          type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction (use if already done).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
