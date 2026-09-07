"""
detector.py
-----------
Stage 5: High-level deepfake detection inference wrapper.

Given a video path or a single image, this module:
  1. Loads all four trained models (spatial, frequency, temporal, fusion).
  2. Extracts frames from the video at a configurable FPS.
  3. Detects faces in each frame using Haar Cascade.
  4. Runs each face crop through the spatial + frequency models.
  5. Builds a proper temporal feature sequence from consecutive frames
     (using the actual frame order, not the training approximation).
  6. Combines all cue scores via the fusion meta-learner.
  7. Aggregates frame-level results into a single video-level verdict.
  8. Prints a detailed, colour-coded report.

Usage
~~~~~
  python detector.py <video_path>
  python detector.py <video_path> --fps 5 --threshold 0.5
  python detector.py <image_path>          # single frame mode
  python detector.py --help
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #
MODELS_DIR      = "models"
SPATIAL_MODEL   = os.path.join(MODELS_DIR, "spatial_model.keras")
FREQUENCY_MODEL = os.path.join(MODELS_DIR, "frequency_model.keras")
TEMPORAL_MODEL  = os.path.join(MODELS_DIR, "temporal_model.keras")
FUSION_MODEL    = os.path.join(MODELS_DIR, "fusion_model.keras")

IMAGE_SIZE      = (224, 224)
SEQ_LEN         = 10           # must match temporal_cue.py setting
FEATURE_DIM     = 1280         # EfficientNetB0 GAP output
DEFAULT_FPS     = 3            # frames to sample per second of video
DEFAULT_THRESH  = 0.5          # P(fake) threshold for FAKE verdict

HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Supported single-image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
# =========================================================================== #


# --------------------------------------------------------------------------- #
#  DeepfakeDetector class
# --------------------------------------------------------------------------- #

class DeepfakeDetector:
    """
    All-in-one deepfake detection pipeline.

    Loads all models once and exposes:
      detect_video(path)  ->  result dict
      detect_image(path)  ->  result dict
    """

    def __init__(self, threshold: float = DEFAULT_THRESH):
        self.threshold = threshold
        self._models   = {}
        self._backbone = None   # EfficientNetB0 feature extractor for temporal
        self._cascade  = None   # Haar face detector

        self._load_models()
        self._load_backbone()
        self._load_cascade()

    # ------------------------------------------------------------------ #
    #  Initialisation helpers
    # ------------------------------------------------------------------ #

    def _load_models(self) -> None:
        """Load all four trained models. Missing models are skipped."""
        specs = [
            ("spatial",   SPATIAL_MODEL),
            ("frequency", FREQUENCY_MODEL),
            ("temporal",  TEMPORAL_MODEL),
            ("fusion",    FUSION_MODEL),
        ]
        print("\n[INFO] Loading models ...")
        for name, path in specs:
            if os.path.exists(path):
                self._models[name] = keras.models.load_model(path)
                print(f"  [OK ] {name:<12} <- {path}")
            else:
                print(f"  [MISS] {name:<12}   {path}")

        if not self._models:
            print("\n[ERROR] No models found. Train the models first.")
            sys.exit(1)

        if "fusion" not in self._models and len(self._models) < 2:
            print("\n[WARN] Fusion model missing and fewer than 2 cue models "
                  "available. Results will be less reliable.")

    def _load_backbone(self) -> None:
        """
        Build an EfficientNetB0 feature extractor for the temporal pipeline.
        Tries to reuse the backbone from the trained spatial model.
        """
        if "spatial" in self._models:
            for layer in self._models["spatial"].layers:
                if "efficientnet" in layer.name.lower():
                    inp = keras.Input(shape=(*IMAGE_SIZE, 3))
                    x   = layer(inp, training=False)
                    x   = keras.layers.GlobalAveragePooling2D()(x)
                    self._backbone = keras.Model(inp, x,
                                                 name="TemporalBackbone")
                    layer.trainable = False
                    print("[INFO] Temporal backbone: reusing spatial EfficientNetB0.")
                    return

        # Fallback to fresh ImageNet weights
        self._backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(*IMAGE_SIZE, 3), pooling="avg"
        )
        self._backbone.trainable = False
        print("[INFO] Temporal backbone: fresh EfficientNetB0 (ImageNet).")

    def _load_cascade(self) -> None:
        """Load the Haar Cascade face detector."""
        if not os.path.isfile(HAAR_XML):
            print(f"[ERROR] Haar Cascade not found: {HAAR_XML}")
            sys.exit(1)
        self._cascade = cv2.CascadeClassifier(HAAR_XML)
        print(f"[INFO] Haar Cascade loaded.")

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _preprocess_spatial(img_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> EfficientNetB0 tensor (1, 224, 224, 3)."""
        img = cv2.resize(img_bgr, IMAGE_SIZE)
        img = img[:, :, ::-1].astype(np.float32)           # BGR->RGB
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img[np.newaxis]

    @staticmethod
    def _preprocess_frequency(img_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> FFT log-magnitude -> MobileNetV2 tensor (1, 224, 224, 3)."""
        img = cv2.resize(img_bgr, IMAGE_SIZE)
        fft_ch = []
        for c in range(3):
            ch   = img[:, :, c].astype(np.float32)
            fsh  = np.fft.fftshift(np.fft.fft2(ch))
            mag  = np.log1p(np.abs(fsh))
            norm = cv2.normalize(mag, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
            fft_ch.append(norm)
        fft_img = np.stack(fft_ch, axis=-1).astype(np.float32)
        fft_img = tf.keras.applications.mobilenet_v2.preprocess_input(fft_img)
        return fft_img[np.newaxis]

    def _extract_feature(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> (FEATURE_DIM,) EfficientNetB0 feature vector."""
        x   = self._preprocess_spatial(img_bgr)             # (1, 224, 224, 3)
        return self._backbone.predict(x, verbose=0)[0]      # (FEATURE_DIM,)

    def _detect_faces(self, frame: np.ndarray) -> list:
        """
        Run Haar Cascade on a frame and return a list of face crops (BGR).
        Returns [] if no face is detected.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        crops = []
        for (x, y, w, h) in faces:
            crops.append(frame[y:y+h, x:x+w])
        return crops

    # ------------------------------------------------------------------ #
    #  Per-frame prediction
    # ------------------------------------------------------------------ #

    def _predict_spatial(self, face_bgr: np.ndarray) -> float:
        """Return P(fake) from the spatial model."""
        if "spatial" not in self._models:
            return 0.5
        x = self._preprocess_spatial(face_bgr)
        return float(self._models["spatial"].predict(x, verbose=0)[0, 0])

    def _predict_frequency(self, face_bgr: np.ndarray) -> float:
        """Return P(fake) from the frequency model."""
        if "frequency" not in self._models:
            return 0.5
        x = self._preprocess_frequency(face_bgr)
        return float(self._models["frequency"].predict(x, verbose=0)[0, 0])

    def _predict_temporal(self, feat_sequence: list) -> float:
        """
        Return P(fake) from the temporal model given a list of feature vectors.

        feat_sequence : list of np.ndarray, each shape (FEATURE_DIM,)
                        Built from consecutive detected frames.
        """
        if "temporal" not in self._models or not feat_sequence:
            return 0.5

        # Pad or truncate to SEQ_LEN
        seq = feat_sequence[:SEQ_LEN]
        if len(seq) < SEQ_LEN:
            pad = [np.zeros(FEATURE_DIM)] * (SEQ_LEN - len(seq))
            seq = seq + pad

        seq_arr = np.stack(seq, axis=0)[np.newaxis].astype(np.float32)
        return float(self._models["temporal"].predict(seq_arr, verbose=0)[0, 0])

    def _predict_fusion(self, p_s: float, p_f: float, p_t: float) -> float:
        """Combine three cue scores via the fusion meta-learner."""
        if "fusion" not in self._models:
            # Fallback: weighted average
            return 0.5 * p_s + 0.25 * p_f + 0.25 * p_t

        # Collect available cue scores in the order the fusion model expects.
        # The fusion model was trained with columns [spatial, frequency, temporal]
        # (only the columns for models that existed at training time).
        available_probs = []
        for cue, p in [("spatial", p_s), ("frequency", p_f), ("temporal", p_t)]:
            if cue in self._models:
                available_probs.append(p)

        cue_vec = np.array(available_probs, dtype=np.float32)[np.newaxis]
        return float(self._models["fusion"].predict(cue_vec, verbose=0)[0, 0])

    # ------------------------------------------------------------------ #
    #  Video detection
    # ------------------------------------------------------------------ #

    def detect_video(self, video_path: str,
                     target_fps: int = DEFAULT_FPS) -> dict:
        """
        Analyse a video file and return a full detection result.

        Parameters
        ----------
        video_path : str – path to the video file
        target_fps : int – frames to sample per second

        Returns
        -------
        dict with keys:
            verdict, fusion_prob, confidence,
            spatial_prob, frequency_prob, temporal_prob,
            frames_analysed, faces_detected, per_frame_results
        """
        if not os.path.isfile(video_path):
            return {"error": f"File not found: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        native_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(native_fps / target_fps)))

        print(f"\n[INFO] Video  : {os.path.basename(video_path)}")
        print(f"       FPS    : {native_fps:.1f}  "
              f"Frames: {total_frames}  "
              f"Sample every: {frame_interval}")

        # Accumulators
        all_spatial    = []
        all_frequency  = []
        feat_sequence  = []   # temporal feature vectors (from consecutive frames)
        per_frame      = []
        frames_analysed = 0
        faces_detected  = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames_analysed += 1
                crops = self._detect_faces(frame)

                if crops:
                    # Use the first (largest-area) face for simplicity
                    face = crops[0]
                    faces_detected += 1

                    p_s = self._predict_spatial(face)
                    p_f = self._predict_frequency(face)
                    feat_sequence.append(self._extract_feature(face))

                    all_spatial.append(p_s)
                    all_frequency.append(p_f)
                    per_frame.append({
                        "frame_idx": frame_idx,
                        "spatial":   round(p_s, 4),
                        "frequency": round(p_f, 4),
                    })

            frame_idx += 1

        cap.release()

        if not all_spatial:
            print("[WARN] No faces detected in any frame.")
            return {
                "verdict":        "UNCERTAIN",
                "fusion_prob":    0.5,
                "confidence":     0.0,
                "frames_analysed": frames_analysed,
                "faces_detected":  0,
                "error":          "No faces detected.",
            }

        # Aggregate per-frame spatial + frequency
        p_s_agg = float(np.mean(all_spatial))
        p_f_agg = float(np.mean(all_frequency))

        # Temporal prediction on the collected sequence
        p_t_agg = self._predict_temporal(feat_sequence)

        # Fusion
        p_fuse = self._predict_fusion(p_s_agg, p_f_agg, p_t_agg)

        verdict    = "FAKE" if p_fuse >= self.threshold else "REAL"
        confidence = abs(p_fuse - 0.5) * 2   # 0=uncertain, 1=fully confident

        return {
            "verdict":          verdict,
            "fusion_prob":      round(p_fuse, 4),
            "confidence":       round(confidence, 4),
            "spatial_prob":     round(p_s_agg, 4),
            "frequency_prob":   round(p_f_agg, 4),
            "temporal_prob":    round(p_t_agg, 4),
            "frames_analysed":  frames_analysed,
            "faces_detected":   faces_detected,
            "per_frame_results": per_frame,
        }

    # ------------------------------------------------------------------ #
    #  Single image detection
    # ------------------------------------------------------------------ #

    def detect_image(self, image_path: str) -> dict:
        """
        Analyse a single image file (face crop or full photo).

        If the image contains a face, crops it first.
        Falls back to the full image if no face is detected.
        """
        if not os.path.isfile(image_path):
            return {"error": f"File not found: {image_path}"}

        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Cannot read image: {image_path}"}

        crops = self._detect_faces(img)
        face  = crops[0] if crops else img   # use full image if no face found

        p_s    = self._predict_spatial(face)
        p_f    = self._predict_frequency(face)
        # Single image: repeat feature SEQ_LEN times for temporal
        feat   = self._extract_feature(face)
        p_t    = self._predict_temporal([feat] * SEQ_LEN)
        p_fuse = self._predict_fusion(p_s, p_f, p_t)

        verdict    = "FAKE" if p_fuse >= self.threshold else "REAL"
        confidence = abs(p_fuse - 0.5) * 2

        return {
            "verdict":       verdict,
            "fusion_prob":   round(p_fuse, 4),
            "confidence":    round(confidence, 4),
            "spatial_prob":  round(p_s, 4),
            "frequency_prob": round(p_f, 4),
            "temporal_prob": round(p_t, 4),
            "face_detected": bool(crops),
        }


# --------------------------------------------------------------------------- #
#  Pretty-print result
# --------------------------------------------------------------------------- #

def print_result(result: dict, input_path: str) -> None:
    """Print a formatted detection report to the console."""

    if "error" in result and result.get("verdict") not in ("FAKE", "REAL"):
        print(f"\n[ERROR] {result['error']}")
        return

    verdict    = result.get("verdict", "UNKNOWN")
    fusion_p   = result.get("fusion_prob", 0.5)
    confidence = result.get("confidence", 0.0)

    # Build verdict banner
    if verdict == "FAKE":
        banner = "** DEEPFAKE DETECTED **"
    elif verdict == "REAL":
        banner = "** AUTHENTIC VIDEO **"
    else:
        banner = "** UNCERTAIN **"

    print("\n" + "=" * 60)
    print(f"  DEEPFAKE DETECTION RESULT")
    print("=" * 60)
    print(f"  Input      : {os.path.basename(input_path)}")

    # Video-specific fields
    if "frames_analysed" in result:
        print(f"  Frames     : {result['frames_analysed']} analysed  "
              f"| {result['faces_detected']} with faces")

    print("-" * 60)
    print(f"  Spatial    probability : {result.get('spatial_prob',   'N/A'):.4f}")
    print(f"  Frequency  probability : {result.get('frequency_prob', 'N/A'):.4f}")
    print(f"  Temporal   probability : {result.get('temporal_prob',  'N/A'):.4f}")
    print("-" * 60)
    print(f"  FUSION     probability : {fusion_p:.4f}")
    print(f"  Confidence             : {confidence * 100:.1f}%")
    print("=" * 60)
    print(f"  VERDICT  :  {banner}")
    print("=" * 60)

    # Per-frame breakdown (show first 10 frames only)
    pf = result.get("per_frame_results", [])
    if pf:
        print(f"\n  Per-frame breakdown (first {min(10, len(pf))} frames):")
        print(f"  {'Frame':<8} {'Spatial':>10} {'Frequency':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*12}")
        for fr in pf[:10]:
            print(f"  {fr['frame_idx']:<8} "
                  f"{fr['spatial']:>10.4f} "
                  f"{fr['frequency']:>12.4f}")
        if len(pf) > 10:
            print(f"  ... and {len(pf) - 10} more frames.")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION - Inference  (Stage 5)")
    print("=" * 60)
    print(f"  Input     : {args.input}")
    print(f"  Threshold : {args.threshold}")
    if not args.image_mode:
        print(f"  Sample FPS: {args.fps}")
    print("=" * 60)

    t_start = time.time()

    # 1. Build detector (loads all models)
    detector = DeepfakeDetector(threshold=args.threshold)

    # 2. Detect
    ext = os.path.splitext(args.input)[1].lower()

    if args.image_mode or ext in IMAGE_EXTS:
        print("\n[INFO] Running single-image detection ...")
        result = detector.detect_image(args.input)
    else:
        print("\n[INFO] Running video detection ...")
        result = detector.detect_video(args.input, target_fps=args.fps)

    # 3. Report
    print_result(result, args.input)

    elapsed = time.time() - t_start
    print(f"\n[INFO] Analysis completed in {elapsed:.1f}s")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: Run deepfake detection on a video or image."
    )
    parser.add_argument(
        "input",
        help="Path to the video or image file to analyse."
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"Frames per second to sample from video (default: {DEFAULT_FPS})."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESH,
        help=f"P(fake) threshold for FAKE verdict (default: {DEFAULT_THRESH})."
    )
    parser.add_argument(
        "--image-mode", action="store_true",
        help="Force single-image mode even for video-extension files."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
