# Deep Learning Based Multi-Cue Deepfake Detection

> **College Project** | Python 3.10.11 · TensorFlow 2.16.2 · OpenCV 4.10.0

---

## What This Project Does

This system detects whether a video has been manipulated using deepfake techniques by combining **three complementary cues**:

| Cue | What it captures |
|-----|-----------------|
| **Spatial** | Per-frame visual artefacts (texture, colour, blending errors) |
| **Frequency** | GAN-generated patterns visible in the DCT / FFT domain |
| **Temporal** | Unnatural motion between frames (blinking, head-pose jitter) |

A final **fusion module** weighs all three cues to produce a single `REAL / FAKE` decision with a confidence score.

The primary dataset is **FaceForensics++** (FF++).

---

## Folder Structure

```
deepfake detection/
│
├── dataset/
│   ├── raw/
│   │   ├── real/          ← Place your FF++ original videos here
│   │   └── fake/          ← Place your FF++ manipulated videos here
│   │
│   ├── frames/
│   │   ├── real/          ← Auto-generated: extracted frames (real)
│   │   └── fake/          ← Auto-generated: extracted frames (fake)
│   │
│   └── faces/
│       ├── real/          ← Auto-generated: cropped faces 224×224 (real)
│       └── fake/          ← Auto-generated: cropped faces 224×224 (fake)
│
├── models/                ← Trained model weights (auto-generated)
│
├── test_video.py          ← Stage 1: Inspect a single video file
├── preprocess.py          ← Stage 1: Extract frames + crop faces
├── train_spatial.py       ← Stage 2: Train spatial CNN (TODO)
├── frequency_cue.py       ← Stage 3: Frequency-domain analysis (TODO)
├── temporal_cue.py        ← Stage 3: Temporal modelling (TODO)
├── fusion.py              ← Stage 4: Multi-cue fusion (TODO)
├── detector.py            ← Stage 5: Inference wrapper (TODO)
├── app.py                 ← Stage 6: Web demo (TODO)
└── requirements.txt
```

---

## Dataset Setup: FaceForensics++

FaceForensics++ requires academic access. You can request it here:
**https://github.com/ondyari/FaceForensics**

Once you have the dataset, place the videos as follows:

| Video type | Where to put them |
|------------|-------------------|
| Original / real videos | `dataset/raw/real/` |
| Manipulated / fake videos | `dataset/raw/fake/` |

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`

**Example layout after copying:**
```
dataset/raw/real/000.mp4
dataset/raw/real/001.mp4
dataset/raw/fake/000_Deepfakes.mp4
dataset/raw/fake/001_Face2Face.mp4
```

> The preprocessing script labels each video automatically based on which
> folder it lives in — no manual labelling required.

---

## Stage 1 – How to Run

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.16.2 should already be in your environment.  
> If you need to recreate a fresh virtual environment, also run:  
> `pip install tensorflow==2.16.2`

---

### 1. Test a single video

```bash
python test_video.py dataset/raw/real/000.mp4
```

**Expected output:**
```
==================================================
  VIDEO INFO
==================================================
  File       : C:\...\dataset\raw\real\000.mp4
  FPS        : 25.00
  Frames     : 750
  Duration   : 30.00 seconds  (0.50 minutes)
  Resolution : 1920 x 1080 px
==================================================
[OK] Video opened and closed successfully.
```

If you see `[ERROR] Cannot open video`, the file may be corrupt or the
codec is missing. Try re-downloading the video or installing `ffmpeg`.

---

### 2. Run the preprocessing pipeline

```bash
# Default: extract 3 frames per second
python preprocess.py

# Extract 5 frames per second
python preprocess.py --fps 5
```

**Expected output (example with 2 videos):**
```
============================================================
  DEEPFAKE DETECTION – Preprocessing Pipeline  (Stage 1)
============================================================
  Target FPS  : 3
  Face size   : 224 x 224 px
============================================================

[OK] Loaded Haar Cascade from: ...

────────────────────────────────────────────────────────────
  Processing REAL videos from:  dataset/raw/real
────────────────────────────────────────────────────────────
  Found 2 video(s).

  [1/2] 000.mp4
  → Extracting frames …
    Extracted 90 frames  (every 8 native frame(s))
  → Detecting faces in 90 frame(s) …
  → Saved 87 face crop(s) to: dataset/faces/real

  [2/2] 001.mp4
  → Extracting frames …
    ...

────────────────────────────────────────────────────────────
  Processing FAKE videos from:  dataset/raw/fake
────────────────────────────────────────────────────────────
  ...

============================================================
  PIPELINE COMPLETE
============================================================
  Total frames extracted : 180
  Total face crops saved : 174
============================================================
```

**Output files:**
- `dataset/frames/real/<video>_frame<NNNNNN>.jpg`  ← raw extracted frames
- `dataset/frames/fake/<video>_frame<NNNNNN>.jpg`
- `dataset/faces/real/<video>_frame<NNNNNN>_face<NN>.jpg`  ← 224×224 crops
- `dataset/faces/fake/<video>_frame<NNNNNN>_face<NN>.jpg`

---

## Environment

| Package | Version |
|---------|---------|
| Python | 3.10.11 |
| TensorFlow | 2.16.2 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0 |

> ⚠️ Do **not** upgrade NumPy to ≥ 2.x — TensorFlow 2.16.2 is incompatible.

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Project setup, frame extraction, face cropping |
| 2 | 🔲 TODO | Spatial CNN (EfficientNet / Xception) training |
| 3 | 🔲 TODO | Frequency cue (DCT / FFT features) |
| 3 | 🔲 TODO | Temporal cue (LSTM / 3D-CNN) |
| 4 | 🔲 TODO | Multi-cue fusion module |
| 5 | 🔲 TODO | Inference wrapper (`detector.py`) |
| 6 | 🔲 TODO | Web demo (`app.py`) |
