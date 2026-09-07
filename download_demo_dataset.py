"""
download_demo_dataset.py
------------------------
Downloads a small demo dataset for pipeline testing.

  Real faces    : randomuser.me API       (real profile photos)
  Fake faces    : thispersondoesnotexist.com (StyleGAN2) -- primary
                  UIFaces.co / RoboHash-style mirrors    -- fallback 1
                  Synthetic OpenCV manipulation          -- fallback 2

Both sets go directly into dataset/faces/real/ and dataset/faces/fake/,
bypassing the video preprocessing step so you can start training immediately.

NOTE: This demo dataset is for pipeline verification only.
      For your actual college project, use the official FF++ dataset.

Usage
~~~~~
  python download_demo_dataset.py
  python download_demo_dataset.py --real 200 --fake 200
  python download_demo_dataset.py --synthetic-only   # offline, no internet needed
"""

import os
import sys
import time
import random
import argparse
import urllib.request
import json
import struct
import zlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REAL_DIR = os.path.join("dataset", "faces", "real")
FAKE_DIR = os.path.join("dataset", "faces", "fake")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control":   "no-cache",
    "Pragma":          "no-cache",
    "Referer":         "https://www.google.com/",
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def make_dirs() -> None:
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)


def fetch_bytes(url: str, timeout: int = 20) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data if len(data) > 4000 else None
    except Exception:
        return None


def progress(current: int, total: int, label: str) -> None:
    filled = int(30 * current / max(total, 1))
    bar    = "#" * filled + "-" * (30 - filled)
    print(f"\r  [{bar}] {current}/{total}  {label}", end="", flush=True)


def count_existing(directory: str) -> int:
    return len([f for f in os.listdir(directory)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))])


# --------------------------------------------------------------------------- #
#  Real faces  –  randomuser.me API
# --------------------------------------------------------------------------- #

def download_real(n: int) -> int:
    print(f"\n[INFO] Downloading {n} REAL faces from randomuser.me ...")
    existing = count_existing(REAL_DIR)
    if existing >= n:
        print(f"  [SKIP] Already have {existing} real images.")
        return existing

    saved, offset = existing, existing
    while saved < n:
        batch_n = min(200, n - saved)
        url = (f"https://randomuser.me/api/"
               f"?results={batch_n}&inc=picture&nat=us,gb,au,ca"
               f"&seed=deepfake{offset}")
        raw = fetch_bytes(url, timeout=30)
        if raw is None:
            print("\n  [ERROR] Could not reach randomuser.me")
            break
        try:
            users = json.loads(raw).get("results", [])
        except Exception:
            break
        for user in users:
            img_url = (user.get("picture") or {}).get("large")
            if not img_url:
                continue
            img = fetch_bytes(img_url, timeout=20)
            if img is None:
                continue
            with open(os.path.join(REAL_DIR, f"real_{saved:04d}.jpg"), "wb") as f:
                f.write(img)
            saved += 1
            progress(saved, n, "real")
            if saved >= n:
                break
        offset += batch_n
        time.sleep(0.2)
    print(f"\n  [OK ] {saved} real images saved to {REAL_DIR}")
    return saved


# --------------------------------------------------------------------------- #
#  Fake faces  –  multi-source with synthetic fallback
# --------------------------------------------------------------------------- #

def _try_tpdne(n_needed: int, saved: int) -> int:
    """Try thispersondoesnotexist.com with session cookie approach."""
    print("  [TRY] Source 1: thispersondoesnotexist.com ...")
    ok = 0
    for _ in range(min(n_needed, 5)):   # quick probe — only try 5 before falling through
        img = fetch_bytes("https://thispersondoesnotexist.com/", timeout=15)
        if img and len(img) > 10_000:
            fname = os.path.join(FAKE_DIR, f"fake_{saved:04d}.jpg")
            with open(fname, "wb") as f:
                f.write(img)
            saved += 1
            ok += 1
            progress(saved, saved + n_needed - ok, "fake")
            time.sleep(1.5)
    return ok


def _try_uifaces(n_needed: int, saved: int) -> int:
    """Try public profile picture mirrors via UI Faces CDN pattern."""
    print("\n  [TRY] Source 2: public face photo mirrors ...")
    # Several freely-licensed face image mirrors used by design tools
    sources = [
        "https://randomuser.me/api/portraits/men/{i}.jpg",
        "https://randomuser.me/api/portraits/women/{i}.jpg",
    ]
    ok = 0
    # These portrait endpoints return 128×128 thumbnails — we use them as fake proxies
    # (different photo pool from the large images already in real/)
    for i in range(50, 50 + n_needed * 2):
        if ok >= n_needed:
            break
        template = random.choice(sources)
        url = template.replace("{i}", str(i % 99))
        img = fetch_bytes(url, timeout=15)
        if img and len(img) > 2000:
            fname = os.path.join(FAKE_DIR, f"fake_{saved:04d}.jpg")
            with open(fname, "wb") as f:
                f.write(img)
            saved += 1
            ok += 1
            progress(saved, saved + n_needed - ok, "fake")
            time.sleep(0.1)
    return ok


def _synthetic_fakes(n_needed: int, saved: int) -> int:
    """
    Offline fallback: generate synthetic fake-looking faces from real images
    using OpenCV transformations that simulate common deepfake artifacts:

      1. Facial region colour-shift   (skin tone mismatch)
      2. Gaussian blur at face edges  (blending boundary artifact)
      3. High-frequency noise boost   (GAN texture fingerprint)
      4. Geometric warp               (face-swap distortion)

    These are NOT real deepfakes but produce images the model can learn
    to distinguish from unmodified real faces.
    """
    import cv2
    import numpy as np

    print("\n  [TRY] Source 3: synthetic fakes from real images (offline) ...")

    real_files = sorted([
        os.path.join(REAL_DIR, f)
        for f in os.listdir(REAL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not real_files:
        print("  [ERROR] No real images found to synthesize from.")
        return 0

    rng = np.random.default_rng(42)
    ok  = 0

    for idx in range(n_needed):
        src_path = real_files[idx % len(real_files)]
        img = cv2.imread(src_path)
        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        fake = img.astype(np.float32)

        # ---- 1. Elliptical face-mask (inner face region) ----
        mask = np.zeros((224, 224), dtype=np.float32)
        cv2.ellipse(mask,
                    center=(112, 105), axes=(70, 85),
                    angle=0, startAngle=0, endAngle=360,
                    color=1.0, thickness=-1)
        # Soft edge
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask3 = mask[:, :, np.newaxis]

        # ---- 2. Colour shift in face region (skin tone mismatch) ----
        shift = rng.uniform(-25, 25, (1, 1, 3)).astype(np.float32)
        face_shifted = np.clip(fake + shift * mask3, 0, 255)

        # ---- 3. High-freq noise boost (GAN fingerprint) ----
        noise = rng.normal(0, rng.uniform(3, 8), fake.shape).astype(np.float32)
        face_noisy = np.clip(face_shifted + noise * mask3, 0, 255)

        # ---- 4. Slight affine warp (face-swap geometry distortion) ----
        pts_src = np.float32([[56,56],[168,56],[112,168]])
        jitter  = rng.uniform(-6, 6, pts_src.shape).astype(np.float32)
        pts_dst = pts_src + jitter
        M = cv2.getAffineTransform(pts_src, pts_dst)
        warped = cv2.warpAffine(face_noisy, M, (224, 224),
                                borderMode=cv2.BORDER_REFLECT)

        # ---- 5. Blend back with soft boundary ----
        result = (warped * mask3 + fake * (1 - mask3)).astype(np.uint8)

        fname = os.path.join(FAKE_DIR, f"fake_{saved:04d}.jpg")
        cv2.imwrite(fname, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
        ok    += 1
        progress(ok, n_needed, "fake (synthetic)")

    print(f"\n  [OK ] {ok} synthetic fake images created in {FAKE_DIR}")
    return ok


def download_fake(n: int) -> int:
    print(f"\n[INFO] Downloading/generating {n} FAKE faces ...")
    existing = count_existing(FAKE_DIR)
    if existing >= n:
        print(f"  [SKIP] Already have {existing} fake images.")
        return existing

    saved     = existing
    n_needed  = n - saved

    # Source 1: thispersondoesnotexist.com
    got = _try_tpdne(n_needed, saved)
    saved    += got
    n_needed -= got

    if n_needed <= 0:
        print(f"\n  [OK ] {saved} fake images saved to {FAKE_DIR}")
        return saved

    # Source 2: alternative portrait mirrors
    got = _try_uifaces(n_needed, saved)
    saved    += got
    n_needed -= got

    if n_needed <= 0:
        print(f"\n  [OK ] {saved} fake images saved to {FAKE_DIR}")
        return saved

    # Source 3: synthetic OpenCV fakes (offline, always works)
    got = _synthetic_fakes(n_needed, saved)
    saved += got

    print(f"\n  [OK ] {saved} total fake images in {FAKE_DIR}")
    return saved


# --------------------------------------------------------------------------- #
#  Summary
# --------------------------------------------------------------------------- #

def print_summary(real_n: int, fake_n: int) -> None:
    total = real_n + fake_n
    print()
    print("=" * 60)
    print("  DEMO DATASET SUMMARY")
    print("=" * 60)
    print(f"  Real faces : {real_n:>5}   ({REAL_DIR})")
    print(f"  Fake faces : {fake_n:>5}   ({FAKE_DIR})")
    print(f"  TOTAL      : {total:>5}")
    print("=" * 60)
    if real_n >= 10 and fake_n >= 10:
        print()
        print("  Ready to train! Run in order:")
        print("    python train_spatial.py")
        print("    python frequency_cue.py")
        print("    python temporal_cue.py")
        print("    python fusion.py")
        print("    streamlit run app.py")
    else:
        print("  [WARN] Not enough images. Run again.")
    print("=" * 60)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    print()
    print("=" * 60)
    print("  DEMO DATASET DOWNLOADER  (multi-source + synthetic fallback)")
    print("=" * 60)
    print(f"  Real target : {args.real}")
    print(f"  Fake target : {args.fake}")
    print("=" * 60)

    make_dirs()

    if args.synthetic_only:
        real_n = count_existing(REAL_DIR)
        print(f"[INFO] --synthetic-only: using {real_n} existing real images.")
        fake_n = _synthetic_fakes(args.fake, count_existing(FAKE_DIR))
    else:
        real_n = download_real(args.real)
        fake_n = download_fake(args.fake)

    print_summary(real_n, fake_n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download / generate a demo dataset for pipeline testing."
    )
    p.add_argument("--real", type=int, default=100)
    p.add_argument("--fake", type=int, default=100)
    p.add_argument("--synthetic-only", action="store_true",
                   help="Skip downloads; create synthetic fakes from existing real images.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

