"""Face detection, tracking and alignment (report Section 5.1.1, Appendix A.1).

extract() decodes a video at its native frame rate (or reads a still image),
detects faces on the frames the cues need (32 uniform frames for the spatial
and frequency cues, 8 clips of 16 consecutive frames for the temporal cue),
keeps the longest IoU track, smooths its landmarks and returns aligned
224 x 224 BGR crops. RetinaFace replaces the Haar cascade of the earlier
prototype; Haar and MTCNN remain selectable for the ablation in Table 7.5.
"""
import bisect
import os

import cv2
import numpy as np

try:  # must be imported before Keras 3: retina-face switches tf.keras to tf-keras on import
    from retinaface import RetinaFace          # pip install retina-face tf-keras
except ImportError:
    RetinaFace = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ArcFace 5-point template in a 112 x 112 frame: left eye, right eye, nose,
# left mouth corner, right mouth corner (image coordinates).
_TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def make_template(size=224, margin=1.3):
    """Shrink the template toward the centre so the crop keeps a margin
    of `margin` around the face (and with it the blending boundary)."""
    t = (_TEMPLATE_112 - 56.0) / margin + 56.0
    return t * (size / 112.0)


def _five_points(left_eye, right_eye, nose, mouth_a, mouth_b):
    eyes = sorted([left_eye, right_eye], key=lambda p: p[0])
    mouth = sorted([mouth_a, mouth_b], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], nose, mouth[0], mouth[1]], dtype=np.float32)


def _retinaface(img, conf):
    out = RetinaFace.detect_faces(img, threshold=conf, allow_upscaling=False)
    if not isinstance(out, dict):
        return []
    faces = []
    for f in out.values():
        lm = f["landmarks"]
        pts = _five_points(lm["left_eye"], lm["right_eye"], lm["nose"],
                           lm["mouth_left"], lm["mouth_right"])
        faces.append((np.array(f["facial_area"], dtype=np.float32), pts))
    return faces


_cache = {}


def _mtcnn(img, conf):
    if "mtcnn" not in _cache:
        from mtcnn import MTCNN                  # pip install mtcnn (ablation only)
        _cache["mtcnn"] = MTCNN()
    faces = []
    for f in _cache["mtcnn"].detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)):
        if f["confidence"] < conf:
            continue
        x, y, w, h = f["box"]
        k = f["keypoints"]
        pts = _five_points(k["left_eye"], k["right_eye"], k["nose"], k["mouth_left"], k["mouth_right"])
        faces.append((np.array([x, y, x + w, y + h], dtype=np.float32), pts))
    return faces


def _haar(img, conf):
    """Earlier prototype's detector. No landmarks: the 'points' are the two box corners."""
    if "haar" not in _cache:
        _cache["haar"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = _cache["haar"].detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))
    return [(np.array([x, y, x + w, y + h], dtype=np.float32),
             np.array([[x, y], [x + w, y + h]], dtype=np.float32)) for x, y, w, h in boxes]


DETECTORS = {"retinaface": _retinaface, "mtcnn": _mtcnn, "haar": _haar}


def detect(frame_bgr, detector="retinaface", conf=0.9, max_side=640):
    """Return a list of (box[x1, y1, x2, y2], points) for one frame.
    Frames are downscaled to `max_side` for speed; coordinates are mapped back."""
    s = min(1.0, max_side / max(frame_bgr.shape[:2]))
    small = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame_bgr
    return [(box / s, pts / s) for box, pts in DETECTORS[detector](small, conf)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def track(detections, iou_thr=0.5):
    """Greedy IoU linking across frames. `detections[i]` is the output of
    detect() for frame i. Returns the longest track as {frame: points};
    ties (e.g. a still image with two faces) go to the larger face."""
    tracks = []                              # each: {"box": last box, "pts": {frame: points}}
    for i, faces in enumerate(detections):
        used = set()
        for tr in tracks:
            best, best_j = 0.0, None
            for j, (box, _) in enumerate(faces):
                if j not in used and iou(tr["box"], box) > best:
                    best, best_j = iou(tr["box"], box), j
            if best_j is not None and best >= iou_thr:
                tr["box"], tr["pts"][i] = faces[best_j][0], faces[best_j][1]
                used.add(best_j)
        for j, (box, pts) in enumerate(faces):
            if j not in used:
                tracks.append({"box": box, "pts": {i: pts}})
    if not tracks:
        return {}
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return max(tracks, key=lambda t: (len(t["pts"]), area(t["box"])))["pts"]


def smooth(points_by_frame, window=5):
    """Centred moving average of landmarks over frames at most window//2 apart
    (by frame index, so the sparse uniform frames are not averaged together).
    Reduces alignment jitter that would otherwise appear as motion in the
    frame-difference stream."""
    half = window // 2
    frames = sorted(points_by_frame)
    return {f: np.mean([points_by_frame[g] for g in frames if abs(g - f) <= half], axis=0)
            for f in frames}


def interpolate(pts, frames, max_gap):
    """Linear landmark interpolation for frames that were not run through the
    detector, between the nearest tracked frames at most max_gap apart."""
    known, out = sorted(pts), dict(pts)
    for f in frames:
        k = bisect.bisect(known, f)
        if f not in out and 0 < k < len(known) and known[k] - known[k - 1] <= max_gap:
            a, b = known[k - 1], known[k]
            out[f] = pts[a] + (f - a) / (b - a) * (pts[b] - pts[a])
    return out


def align(frame_bgr, pts, size=224, margin=1.3):
    if len(pts) == 2:                        # Haar box: square crop with the same margin
        c = pts.mean(axis=0)
        half = max(pts[1] - pts[0]) * margin / 2
        src = np.float32([[c[0] - half, c[1] - half], [c[0] + half, c[1] - half], [c[0] - half, c[1] + half]])
        M = cv2.getAffineTransform(src, np.float32([[0, 0], [size, 0], [0, size]]))
    else:
        M, _ = cv2.estimateAffinePartial2D(pts, make_template(size, margin), method=cv2.LMEDS)
        if M is None:
            return None
    return cv2.warpAffine(frame_bgr, M, (size, size), borderMode=cv2.BORDER_REFLECT)


def sample_indices(n, n_uniform=32, n_clips=8, clip_len=16):
    """Frame indices used by the cues: uniform frames and evenly spaced clips."""
    uniform = np.unique(np.linspace(0, n - 1, n_uniform).round().astype(int)).tolist() if n > 0 else []
    starts = np.unique(np.linspace(0, max(n - clip_len, 0), n_clips).round().astype(int))
    clips = [list(range(s, s + clip_len)) for s in starts.tolist() if s + clip_len <= n]
    return uniform, clips


def read_frames(path, n_uniform=32, n_clips=8, clip_len=16):
    """Decode only the needed frames. Returns ({frame index: BGR frame}, uniform, clips, n_frames)."""
    if os.path.splitext(path)[1].lower() in IMAGE_EXTS:
        img = cv2.imread(path)
        return ({0: img} if img is not None else {}), [0], [], 1
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    uniform, clips = sample_indices(n, n_uniform, n_clips, clip_len)
    want = set(uniform) | {i for c in clips for i in c}
    frames, i = {}, 0
    while want and i <= max(want):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = fr
        i += 1
    cap.release()
    return frames, uniform, clips, n


def extract(path, detector="retinaface", conf=0.9, size=224, perturb_fn=None,
            n_uniform=32, n_clips=8, clip_len=16, max_side=640, detect_every=1):
    """Aligned face crops for one video or image, plus what was found.
    detect_every > 1 runs the detector on every k-th frame of each clip (and its
    last frame) and interpolates the landmarks in between, which is ~k times faster."""
    frames, uniform, clips, n = read_frames(path, n_uniform, n_clips, clip_len)
    if perturb_fn is not None:               # robustness protocol: whole frame, before detection
        frames = {i: perturb_fn(f) for i, f in frames.items()}
    keys = set(uniform) | {c[j] for c in clips for j in range(0, len(c), detect_every)} | {c[-1] for c in clips}
    order = sorted(keys & frames.keys())
    pts = track([detect(frames[i], detector, conf, max_side) for i in order])
    pts = smooth(interpolate({order[k]: p for k, p in pts.items()}, frames, detect_every))
    crops = {}
    for i, p in pts.items():
        crop = align(frames[i], p, size)
        if crop is not None:
            crops[i] = crop
    return {
        "crops": crops,
        "uniform": [i for i in uniform if i in crops],
        "clips": [c for c in clips if all(i in crops for i in c)],
        "n_frames": n,
        "n_sampled": len(frames),
        "n_found": len(crops),
        "flagged": not crops or 2 * len(crops) < len(frames),   # no face in > half the frames
    }
