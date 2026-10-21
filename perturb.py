"""Frame-level perturbations for the robustness protocol (Section 5.2.4, Appendix A.5).
Applied to whole frames before face detection, as a re-upload would."""
import cv2
import numpy as np

LEVELS = {
    "jpeg":   [90, 70, 50, 30],          # JPEG quality
    "resize": [0.75, 0.5, 0.25],         # down-scale factor, then back up
    "noise":  [5, 10, 20],               # Gaussian sigma on the 0-255 scale
    "blur":   [1, 2, 3],                 # Gaussian blur sigma in pixels
}


def perturb(frame_bgr, kind, level, rng=None):
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "resize":
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (max(1, int(w * level)), max(1, int(h * level))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "noise":
        rng = rng or np.random.default_rng(0)
        noisy = frame_bgr.astype(np.float32) + rng.normal(0, level, frame_bgr.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=level)
    raise ValueError(f"unknown perturbation: {kind}")
