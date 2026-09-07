"""
fusion.py
---------
Stage 4: Multi-cue fusion module.

What This Does
~~~~~~~~~~~~~~
Combines the outputs of all three deepfake detectors into a single,
more robust prediction using a learned meta-learner (stacking ensemble).

              +------------------+
              |  Face crop image |
              +--------+---------+
                       |
          +------------+-------------+
          |            |             |
          v            v             v
    [Spatial]    [Frequency]   [Temporal]
    EfficientNetB0  MobileNetV2  BiLSTM
          |            |             |
          v            v             v
       P(fake)      P(fake)       P(fake)
          |            |             |
          +------------+-------------+
                       |
                       v
              [Meta-Learner MLP]
             Dense(32) -> Dense(16)
                       |
                       v
               Final P(fake)   <-  REAL / FAKE decision

Training Strategy (Stacking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Step 1 - Generate meta-features:
            Run ALL face crops through all 3 models.
            Result: (N_crops, 3) matrix of probability scores.
            Saved to dataset/meta_features.npz

  Step 2 - Train meta-learner:
            Small MLP on the (N_crops, 3) matrix.
            Learns the optimal combination / weighting of the three cues.
            Also computes a weighted-average baseline for comparison.

  Step 3 - Evaluate:
            Compare individual model accuracy vs fusion accuracy.

Output
~~~~~~
  models/fusion_model.keras     <- trained meta-learner
  models/meta_features.npz      <- cached model predictions (reusable)

Usage
~~~~~
  python fusion.py
  python fusion.py --skip-meta-gen     # skip meta-feature generation
  python fusion.py --epochs 50
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
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

print(f"[INFO] TensorFlow : {tf.__version__}")


# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
FACES_DIR       = os.path.join("dataset", "faces")
META_SAVE_PATH  = os.path.join("dataset", "meta_features.npz")

MODELS_DIR      = "models"
SPATIAL_MODEL   = os.path.join(MODELS_DIR, "spatial_model.keras")
FREQUENCY_MODEL = os.path.join(MODELS_DIR, "frequency_model.keras")
TEMPORAL_MODEL  = os.path.join(MODELS_DIR, "temporal_model.keras")
FUSION_MODEL    = os.path.join(MODELS_DIR, "fusion_model.keras")
LOG_PATH        = os.path.join(MODELS_DIR, "fusion_training_log.csv")

# Weights for the simple weighted-average baseline (must sum to 1)
# These can be tuned by hand; the meta-learner will learn optimal weights
BASELINE_WEIGHTS = {
    "spatial":   0.50,   # usually most reliable
    "frequency": 0.25,   # complementary but noisier
    "temporal":  0.25,   # strongest on video-level but weakest per-frame
}

IMAGE_SIZE  = (224, 224)
SEQ_LEN     = 10         # must match temporal_cue.py setting
FEATURE_DIM = 1280       # EfficientNetB0 GAP output

BATCH_SIZE  = 64
VAL_SPLIT   = 0.20
SEED        = 42
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10

THRESHOLD   = 0.5        # probability above which we classify as FAKE
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #

def load_models() -> dict:
    """
    Load all three trained cue models.
    Returns a dict with keys 'spatial', 'frequency', 'temporal'.
    Missing models are logged as warnings (not errors) so partial fusion works.
    """
    models = {}
    specs = [
        ("spatial",   SPATIAL_MODEL),
        ("frequency", FREQUENCY_MODEL),
        ("temporal",  TEMPORAL_MODEL),
    ]

    print("\n[INFO] Loading cue models ...")
    for name, path in specs:
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
            print(f"  [OK ] {name:<12} <- {path}")
        else:
            print(f"  [MISS] {name:<12}    {path}  (not found – will be skipped)")

    if not models:
        print("\n[ERROR] No trained models found in models/")
        print("        Run train_spatial.py, frequency_cue.py, and")
        print("        temporal_cue.py first.")
        sys.exit(1)

    return models


# --------------------------------------------------------------------------- #
#  Image preprocessing helpers
# --------------------------------------------------------------------------- #

def preprocess_spatial(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> EfficientNetB0 input tensor."""
    img = cv2.resize(img_bgr, IMAGE_SIZE)
    img = img[:, :, ::-1].astype(np.float32)           # BGR -> RGB
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def preprocess_frequency(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> FFT log-magnitude spectrum -> MobileNetV2 input."""
    img = cv2.resize(img_bgr, IMAGE_SIZE)

    # 2D FFT per channel (same logic as frequency_cue.py)
    fft_channels = []
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        f_shift = np.fft.fftshift(np.fft.fft2(channel))
        magnitude = np.log1p(np.abs(f_shift))
        magnitude_norm = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        fft_channels.append(magnitude_norm)

    fft_img = np.stack(fft_channels, axis=-1).astype(np.float32)
    fft_img = tf.keras.applications.mobilenet_v2.preprocess_input(fft_img)
    return fft_img


def make_temporal_sequence(img_bgr: np.ndarray,
                            backbone: keras.Model) -> np.ndarray:
    """
    Build a (1, SEQ_LEN, FEATURE_DIM) temporal sequence from a single frame.

    Since we are processing a single image (not a full video clip), we
    extract its EfficientNetB0 feature vector and repeat it SEQ_LEN times.
    This is a deliberate approximation: the temporal model sees a "still"
    clip, which biases it toward the neutral / uncertain range.  The
    meta-learner learns to down-weight temporal confidence in this case.
    """
    img = preprocess_spatial(img_bgr)                   # (224, 224, 3)
    feat = backbone.predict(img[np.newaxis], verbose=0) # (1, FEATURE_DIM)
    feat = feat[0]                                      # (FEATURE_DIM,)

    # Repeat into a SEQ_LEN-length sequence
    seq = np.tile(feat, (SEQ_LEN, 1))                   # (SEQ_LEN, FEATURE_DIM)
    return seq[np.newaxis]                              # (1, SEQ_LEN, FEATURE_DIM)


def build_temporal_backbone(temporal_model: keras.Model) -> keras.Model:
    """
    Extract the EfficientNetB0 feature extractor from the temporal pipeline.

    The temporal model does not embed the backbone; it uses pre-extracted
    features loaded from disk.  So we build a standalone feature extractor
    with fresh EfficientNetB0 weights (or load the spatial backbone if saved).
    """
    if os.path.exists(SPATIAL_MODEL):
        spatial = keras.models.load_model(SPATIAL_MODEL)
        for layer in spatial.layers:
            if "efficientnet" in layer.name.lower():
                backbone_layer = layer
                inp = keras.Input(shape=(*IMAGE_SIZE, 3))
                x   = backbone_layer(inp, training=False)
                x   = layers.GlobalAveragePooling2D()(x)
                extractor = keras.Model(inp, x)
                backbone_layer.trainable = False
                return extractor

    # Fallback
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3), pooling="avg"
    )
    base.trainable = False
    return base


# --------------------------------------------------------------------------- #
#  Step 1: Generate meta-features
# --------------------------------------------------------------------------- #

def generate_meta_features(models: dict) -> tuple:
    """
    Run every face crop through each cue model and collect predictions.

    Returns
    -------
    X_meta : np.ndarray  shape (N, num_cues)  – model probability scores
    y      : np.ndarray  shape (N,)            – ground truth labels (0=real, 1=fake)
    cue_names : list[str]                       – ordered cue names matching columns
    """
    # Determine which models are available
    cue_names = [c for c in ["spatial", "frequency", "temporal"] if c in models]
    print(f"\n[INFO] Generating meta-features using: {cue_names}")

    # Build temporal backbone if needed
    temporal_backbone = None
    if "temporal" in models:
        print("[INFO] Building temporal backbone for single-frame inference ...")
        temporal_backbone = build_temporal_backbone(models["temporal"])

    # Collect (feature_vector, label) pairs
    all_X, all_y = [], []

    for cls_name, label in [("real", 0), ("fake", 1)]:
        cls_dir = os.path.join(FACES_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        files = [f for f in sorted(os.listdir(cls_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not files:
            print(f"  [WARN] No images in {cls_dir}")
            continue

        print(f"\n  {cls_name.upper()}: {len(files)} crops ...")

        for i, fname in enumerate(files):
            path = os.path.join(cls_dir, fname)
            img  = cv2.imread(path)
            if img is None:
                continue

            probs = []

            # Spatial prediction
            if "spatial" in models:
                x = preprocess_spatial(img)[np.newaxis]        # (1, 224, 224, 3)
                p = float(models["spatial"].predict(x, verbose=0)[0, 0])
                probs.append(p)

            # Frequency prediction
            if "frequency" in models:
                x = preprocess_frequency(img)[np.newaxis]      # (1, 224, 224, 3)
                p = float(models["frequency"].predict(x, verbose=0)[0, 0])
                probs.append(p)

            # Temporal prediction (single frame -> repeated sequence)
            if "temporal" in models:
                seq = make_temporal_sequence(img, temporal_backbone)
                p = float(models["temporal"].predict(seq, verbose=0)[0, 0])
                probs.append(p)

            all_X.append(probs)
            all_y.append(label)

            # Progress every 50 images
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(files)} ...")

    if not all_X:
        print("\n[ERROR] No meta-features generated. Check dataset/faces/")
        sys.exit(1)

    X_meta = np.array(all_X, dtype=np.float32)
    y      = np.array(all_y, dtype=np.float32)

    print(f"\n[INFO] Meta-feature matrix: {X_meta.shape}  labels: {y.shape}")
    print(f"       Cues: {cue_names}")

    # Save so we can re-run training without re-running all 3 models
    np.savez(META_SAVE_PATH, X=X_meta, y=y, cue_names=np.array(cue_names))
    print(f"[OK ] Meta-features saved to: {META_SAVE_PATH}")

    return X_meta, y, cue_names


# --------------------------------------------------------------------------- #
#  Weighted average baseline
# --------------------------------------------------------------------------- #

def weighted_average_baseline(X_meta: np.ndarray, y: np.ndarray,
                               cue_names: list) -> float:
    """
    Compute accuracy of a simple weighted average of the cue probabilities.
    Prints per-cue accuracy and ensemble accuracy for comparison.
    """
    print("\n" + "=" * 60)
    print("  WEIGHTED AVERAGE BASELINE")
    print("=" * 60)

    # Per-cue individual accuracies
    print("  Individual cue accuracy:")
    for i, cue in enumerate(cue_names):
        preds = (X_meta[:, i] >= THRESHOLD).astype(float)
        acc   = np.mean(preds == y)
        print(f"    {cue:<12} : {acc:.4f}")

    # Weighted ensemble
    weights = np.array([
        BASELINE_WEIGHTS.get(c, 1.0 / len(cue_names)) for c in cue_names
    ], dtype=np.float32)
    weights /= weights.sum()   # normalise

    ensemble_prob = X_meta @ weights
    ensemble_pred = (ensemble_prob >= THRESHOLD).astype(float)
    ensemble_acc  = np.mean(ensemble_pred == y)

    print(f"\n  Ensemble weights  : { {c: round(float(w), 3) for c,w in zip(cue_names, weights)} }")
    print(f"  Ensemble accuracy : {ensemble_acc:.4f}")
    print("=" * 60)

    return ensemble_acc


# --------------------------------------------------------------------------- #
#  Step 2: Build meta-learner
# --------------------------------------------------------------------------- #

def build_fusion_model(n_cues: int) -> keras.Model:
    """
    Tiny MLP meta-learner.

    Input  : (n_cues,) – probability scores from each cue model
    Output : P(fake)

    This is intentionally small because:
      - The inputs are already high-level semantic probabilities.
      - We have few training examples at this level (one per face crop).
      - A small model avoids overfitting.
    """
    inputs  = keras.Input(shape=(n_cues,), name="cue_scores")
    x       = layers.Dense(32, activation="relu", name="fc1")(inputs)
    x       = layers.Dense(16, activation="relu", name="fc2")(x)
    x       = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="FusionMetaLearner")
    print(f"[INFO] Fusion meta-learner  –  {model.count_params():,} params")
    return model


# --------------------------------------------------------------------------- #
#  Step 3: Train meta-learner
# --------------------------------------------------------------------------- #

def split(X, y, val_split, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    val_n = max(1, int(len(X) * val_split))
    return X[val_n:], y[val_n:], X[:val_n], y[:val_n]


def train_fusion(model: keras.Model,
                 X_meta: np.ndarray, y: np.ndarray,
                 epochs: int) -> None:
    """Train the meta-learner MLP."""
    X_train, y_train, X_val, y_val = split(X_meta, y, VAL_SPLIT, SEED)

    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")

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
        ModelCheckpoint(FUSION_MODEL, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(LOG_PATH, append=False),
    ]

    print("\n" + "=" * 60)
    print("  Training fusion meta-learner ...")
    print("=" * 60)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    return X_val, y_val


def evaluate_fusion(X_val: np.ndarray, y_val: np.ndarray,
                    cue_names: list, baseline_acc: float) -> None:
    """Compare meta-learner vs baseline."""
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    if not os.path.exists(FUSION_MODEL):
        print("[WARN] Fusion model not found.")
        return

    best = keras.models.load_model(FUSION_MODEL)
    loss, acc, auc = best.evaluate(X_val, y_val, verbose=0)

    print(f"  Weighted avg baseline : {baseline_acc:.4f}")
    print(f"  Meta-learner accuracy : {acc:.4f}   AUC: {auc:.4f}")
    print(f"  Improvement           : {(acc - baseline_acc) * 100:+.2f}%")
    print("=" * 60)
    print(f"[OK ] Fusion model : {FUSION_MODEL}")
    print(f"[OK ] Log          : {LOG_PATH}")


# --------------------------------------------------------------------------- #
#  Inference helper (used by detector.py in Stage 5)
# --------------------------------------------------------------------------- #

def predict_fusion(img_bgr: np.ndarray,
                   spatial_model:   keras.Model,
                   frequency_model: keras.Model,
                   temporal_model:  keras.Model,
                   fusion_model:    keras.Model,
                   temporal_backbone: keras.Model) -> dict:
    """
    Run a single face crop through all models and return a result dict.

    Returns
    -------
    dict with keys:
        'spatial_prob'  : float   – P(fake) from spatial model
        'frequency_prob': float   – P(fake) from frequency model
        'temporal_prob' : float   – P(fake) from temporal model
        'fusion_prob'   : float   – P(fake) from meta-learner
        'verdict'       : str     – 'FAKE' or 'REAL'
        'confidence'    : float   – distance from 0.5 decision boundary
    """
    probs = []

    if spatial_model is not None:
        x = preprocess_spatial(img_bgr)[np.newaxis]
        p_s = float(spatial_model.predict(x, verbose=0)[0, 0])
        probs.append(p_s)
    else:
        p_s = 0.5

    if frequency_model is not None:
        x = preprocess_frequency(img_bgr)[np.newaxis]
        p_f = float(frequency_model.predict(x, verbose=0)[0, 0])
        probs.append(p_f)
    else:
        p_f = 0.5

    if temporal_model is not None and temporal_backbone is not None:
        seq  = make_temporal_sequence(img_bgr, temporal_backbone)
        p_t  = float(temporal_model.predict(seq, verbose=0)[0, 0])
        probs.append(p_t)
    else:
        p_t = 0.5

    # Fusion prediction
    cue_vec = np.array(probs, dtype=np.float32)[np.newaxis]
    p_fuse  = float(fusion_model.predict(cue_vec, verbose=0)[0, 0])

    verdict    = "FAKE" if p_fuse >= THRESHOLD else "REAL"
    confidence = abs(p_fuse - 0.5) * 2   # 0 = uncertain, 1 = fully confident

    return {
        "spatial_prob":   p_s,
        "frequency_prob": p_f,
        "temporal_prob":  p_t,
        "fusion_prob":    p_fuse,
        "verdict":        verdict,
        "confidence":     confidence,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION - Fusion Module  (Stage 4)")
    print("=" * 60)
    print(f"  Strategy       : Stacking (meta-learner MLP)")
    print(f"  Baseline       : Weighted average")
    print(f"  Meta-learner   : Dense(32) -> Dense(16) -> Dense(1)")
    print(f"  Epochs         : {args.epochs}")
    print("=" * 60)

    # 1. Load all cue models
    models = load_models()

    # 2. Generate meta-features (or load from cache)
    if args.skip_meta_gen and os.path.exists(META_SAVE_PATH):
        print(f"\n[INFO] Loading cached meta-features from {META_SAVE_PATH}")
        data      = np.load(META_SAVE_PATH, allow_pickle=True)
        X_meta    = data["X"]
        y         = data["y"]
        cue_names = list(data["cue_names"])
        print(f"[INFO] Loaded: {X_meta.shape}  cues: {cue_names}")
    else:
        X_meta, y, cue_names = generate_meta_features(models)

    if len(X_meta) < 10:
        print("\n[ERROR] Not enough meta-feature samples to train fusion.")
        print("        Make sure you have face crops and trained cue models.")
        sys.exit(1)

    # 3. Weighted average baseline (no training needed)
    baseline_acc = weighted_average_baseline(X_meta, y, cue_names)

    # 4. Build and train meta-learner
    print("\n" + "=" * 60)
    print("  TRAINING META-LEARNER")
    print("=" * 60)

    fusion_model = build_fusion_model(n_cues=len(cue_names))

    X_val, y_val = train_fusion(fusion_model, X_meta, y, epochs=args.epochs)

    # 5. Final comparison
    evaluate_fusion(X_val, y_val, cue_names, baseline_acc)

    print("\n[DONE] Stage 4 complete.")
    print(f"       Fusion model : {FUSION_MODEL}")
    print(f"       Next step    : python detector.py <video_path>")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: Train multi-cue fusion meta-learner."
    )
    parser.add_argument("--epochs",         type=int, default=EPOCHS,
                        help=f"Meta-learner training epochs (default: {EPOCHS})")
    parser.add_argument("--skip-meta-gen",  action="store_true",
                        help="Load cached meta-features instead of regenerating.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
