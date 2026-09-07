"""
preprocess.py
-------------
Stage 1: Frame extraction + face cropping pipeline.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Scans dataset/raw/real/ and dataset/raw/fake/ for video files.
2. Determines the label (real / fake) from the folder name.
3. Extracts frames at a configurable rate (default: 3 FPS).
4. Saves extracted frames to dataset/frames/real/ or dataset/frames/fake/.
5. Detects faces in each frame using OpenCV Haar Cascade.
6. Crops each detected face, resizes it to 224x224.
7. Saves cropped faces to dataset/faces/real/ or dataset/faces/fake/.

NOTE: The Haar Cascade is used ONLY for face localisation.
      The actual deepfake classification will be done by a deep-learning
      model in later stages.

Usage:
    python preprocess.py [--fps 3]

Arguments:
    --fps   Frames to extract per second of video (default: 3)

Example:
    python preprocess.py
    python preprocess.py --fps 5
"""

import os
import sys
import argparse
import cv2


# --------------------------------------------------------------------------- #
# Paths  –  all relative to the project root
# --------------------------------------------------------------------------- #
RAW_REAL_DIR   = os.path.join("dataset", "raw",    "real")
RAW_FAKE_DIR   = os.path.join("dataset", "raw",    "fake")
FRAMES_REAL    = os.path.join("dataset", "frames", "real")
FRAMES_FAKE    = os.path.join("dataset", "frames", "fake")
FACES_REAL     = os.path.join("dataset", "faces",  "real")
FACES_FAKE     = os.path.join("dataset", "faces",  "fake")

# Face size that every cropped face will be resized to
FACE_SIZE = (224, 224)

# Video file extensions that will be recognised
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Path to OpenCV's built-in frontal-face Haar Cascade XML file
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def ensure_dirs(*dirs: str) -> None:
    """Create output directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def collect_videos(directory: str) -> list[str]:
    """
    Return a sorted list of video file paths found directly inside `directory`.
    Ignores sub-directories and non-video files.
    """
    if not os.path.isdir(directory):
        print(f"  [WARN] Directory does not exist: {directory}  – skipping.")
        return []

    videos = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(os.path.join(directory, filename))
    return videos


# --------------------------------------------------------------------------- #
# Core processing functions
# --------------------------------------------------------------------------- #

def extract_frames(video_path: str, output_dir: str, target_fps: int) -> list[str]:
    """
    Extract frames from `video_path` at `target_fps` and save them as JPEGs.

    Parameters
    ----------
    video_path  : str  – full path to the source video file.
    output_dir  : str  – folder where extracted frame images will be saved.
    target_fps  : int  – how many frames to extract per second of video.

    Returns
    -------
    List of paths to the saved frame images.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path} – skipping.")
        return []

    # Get native video FPS so we know the sampling interval
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        print(f"    [WARN] Could not read FPS for {video_path}. Defaulting to 25.")
        native_fps = 25.0

    # How many native frames to skip between each extracted frame
    frame_interval = max(1, int(round(native_fps / target_fps)))

    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    frame_idx   = 0   # current frame position in the video
    saved_count = 0   # how many frames have been saved

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Save this frame only if it falls on our sampling interval
        if frame_idx % frame_interval == 0:
            filename   = f"{video_name}_frame{frame_idx:06d}.jpg"
            save_path  = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_paths.append(save_path)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"    Extracted {saved_count} frames  (every {frame_interval} native frame(s))")
    return saved_paths


def detect_and_crop_faces(
    frame_paths: list[str],
    output_dir:  str,
    face_cascade: cv2.CascadeClassifier,
) -> int:
    """
    Detect faces in each frame image, crop them, resize to FACE_SIZE,
    and save them to `output_dir`.

    Parameters
    ----------
    frame_paths  : list[str]          – paths to extracted frame images.
    output_dir   : str                – folder where face crops will be saved.
    face_cascade : CascadeClassifier  – loaded Haar Cascade detector.

    Returns
    -------
    Total number of face crops saved.
    """
    total_faces = 0

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"      [WARN] Could not read frame image: {frame_path}")
            continue

        # Convert to greyscale for the Haar detector (faster, works well)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # scaleFactor: how much the image size is reduced at each image scale
        # minNeighbors: how many neighbours each rectangle should retain
        # minSize: minimum face size in pixels
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            # No face found – skip this frame (not a crash, just a warning)
            continue

        # Build a base filename from the frame path (without extension)
        frame_basename = os.path.splitext(os.path.basename(frame_path))[0]

        # Crop and save each detected face
        # (a single frame can contain multiple faces – we handle all of them)
        for face_idx, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y : y + h, x : x + w]

            # Resize to the standard 224x224 input size
            face_resized = cv2.resize(face_crop, FACE_SIZE)

            # Build output filename: <video>_<frame>_face<N>.jpg
            save_name = f"{frame_basename}_face{face_idx:02d}.jpg"
            save_path = os.path.join(output_dir, save_name)

            cv2.imwrite(save_path, face_resized)
            total_faces += 1

    return total_faces


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #

def run_pipeline(target_fps: int) -> None:
    """
    Main preprocessing pipeline.
    Processes real videos first, then fake videos.
    """

    # ------------------------------------------------------------------ #
    # 0. Sanity checks & directory setup
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION – Preprocessing Pipeline  (Stage 1)")
    print("=" * 60)
    print(f"  Target FPS  : {target_fps}")
    print(f"  Face size   : {FACE_SIZE[0]} x {FACE_SIZE[1]} px")
    print("=" * 60 + "\n")

    # Create all output directories
    ensure_dirs(FRAMES_REAL, FRAMES_FAKE, FACES_REAL, FACES_FAKE)

    # Load the Haar Cascade once (reused for every frame)
    if not os.path.isfile(HAAR_CASCADE_PATH):
        print(f"[ERROR] Haar Cascade XML not found at: {HAAR_CASCADE_PATH}")
        print("        Please reinstall opencv-python.")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    print(f"[OK] Loaded Haar Cascade from:\n     {HAAR_CASCADE_PATH}\n")

    # ------------------------------------------------------------------ #
    # 1. Define which source dirs map to which label / output dirs
    # ------------------------------------------------------------------ #
    sources = [
        {
            "label":      "REAL",
            "video_dir":  RAW_REAL_DIR,
            "frames_dir": FRAMES_REAL,
            "faces_dir":  FACES_REAL,
        },
        {
            "label":      "FAKE",
            "video_dir":  RAW_FAKE_DIR,
            "frames_dir": FRAMES_FAKE,
            "faces_dir":  FACES_FAKE,
        },
    ]

    # ------------------------------------------------------------------ #
    # 2. Process each category
    # ------------------------------------------------------------------ #
    grand_total_frames = 0
    grand_total_faces  = 0

    for source in sources:
        label      = source["label"]
        video_dir  = source["video_dir"]
        frames_dir = source["frames_dir"]
        faces_dir  = source["faces_dir"]

        print(f"  Processing {label} videos from:  {video_dir}")
        print("-" * 60)

        videos = collect_videos(video_dir)

        if not videos:
            print(f"  No videos found in {video_dir}.\n")
            continue

        print(f"  Found {len(videos)} video(s).\n")

        for v_idx, video_path in enumerate(videos, start=1):
            video_name = os.path.basename(video_path)
            print(f"  [{v_idx}/{len(videos)}] {video_name}")

            # -- Step A: Extract frames -----------------------------------
            print(f"  >> Extracting frames ...")
            frame_paths = extract_frames(video_path, frames_dir, target_fps)

            if not frame_paths:
                print(f"  >> No frames extracted. Skipping face detection.\n")
                continue

            grand_total_frames += len(frame_paths)

            # -- Step B: Detect & crop faces ------------------------------
            print(f"  >> Detecting faces in {len(frame_paths)} frame(s) ...")
            n_faces = detect_and_crop_faces(frame_paths, faces_dir, face_cascade)

            if n_faces == 0:
                print(f"  >> No faces detected in any frame.")
            else:
                print(f"  >> Saved {n_faces} face crop(s) to: {faces_dir}")

            grand_total_faces += n_faces
            print()

    # ------------------------------------------------------------------ #
    # 3. Final summary
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total frames extracted : {grand_total_frames}")
    print(f"  Total face crops saved : {grand_total_faces}")
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames and crop faces from deepfake videos."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Number of frames to extract per second of video (default: 3).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(target_fps=args.fps)
