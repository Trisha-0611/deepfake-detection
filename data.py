"""Video lists, grouped folds and prediction files shared by the training scripts."""
import csv
import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold


def load_videos(set_dir, split=None, exclude_method=None):
    """Extracted, non-flagged videos of a data set (see prepare_data.py)."""
    videos = []
    with open(os.path.join(set_dir, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            vdir = os.path.join(set_dir, "crops", r["video_id"])
            meta = os.path.join(vdir, "frames.json")
            if (split and r["split"] != split) or r["method"] == exclude_method or not os.path.exists(meta):
                continue
            with open(meta) as m:
                r.update(json.load(m))
            if not r["flagged"]:
                videos.append({**r, "label": int(r["label"]), "dir": vdir, "set_dir": set_dir})
    return videos


def fold_split(videos, k, n_splits=5):
    """Fold k of a 5-fold GroupKFold over FF++ pairs: (train videos, held-out videos)."""
    groups = [v["group"] for v in videos]
    tr, te = list(GroupKFold(n_splits).split(np.zeros(len(videos)), groups=groups))[k]
    return [videos[i] for i in tr], [videos[i] for i in te]


def run_name(train_set, exclude_method=None):
    return os.path.basename(os.path.normpath(train_set)) + (f"_no{exclude_method}" if exclude_method else "")


def write_preds(path, videos, probs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "method", "prob"])
        w.writerows((v["video_id"], v["label"], v["method"], float(p)) for v, p in zip(videos, probs))


def class_weights(videos):
    """Weight the real class by n_fake / n_real (= 4 on the full FF++ split)."""
    n_fake = sum(v["label"] for v in videos)
    return {0: n_fake / max(1, len(videos) - n_fake), 1: 1.0}
