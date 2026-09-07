"""
test_video.py
-------------
Stage 1 utility: open a video file and print basic metadata.
Run this first to confirm that OpenCV can read your videos before
running the full preprocessing pipeline.

Usage:
    python test_video.py <path_to_video>

Example:
    python test_video.py dataset/raw/real/000.mp4
"""

import sys
import os
import cv2


def test_video(video_path: str) -> None:
    """
    Open a video file and print its metadata.

    Parameters
    ----------
    video_path : str
        Path to the video file to inspect.
    """

    # ------------------------------------------------------------------ #
    # 1.  Check that the file actually exists on disk
    # ------------------------------------------------------------------ #
    if not os.path.isfile(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2.  Try to open the video with OpenCV
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open the video: {video_path}")
        print("        Make sure the file is a valid video and that the")
        print("        required codecs are installed on your system.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3.  Read metadata from the VideoCapture object
    # ------------------------------------------------------------------ #
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Duration in seconds; guard against division by zero
    duration_sec = (frame_count / fps) if fps > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 4.  Print a human-readable summary
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("  VIDEO INFO")
    print("=" * 50)
    print(f"  File       : {os.path.abspath(video_path)}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Duration   : {duration_sec:.2f} seconds  ({duration_sec / 60:.2f} minutes)")
    print(f"  Resolution : {width} x {height} px")
    print("=" * 50)

    # ------------------------------------------------------------------ #
    # 5.  Release the capture object
    # ------------------------------------------------------------------ #
    cap.release()
    print("[OK] Video opened and closed successfully.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path_to_video>")
        print("Example: python test_video.py dataset/raw/real/000.mp4")
        sys.exit(1)

    test_video(sys.argv[1])
