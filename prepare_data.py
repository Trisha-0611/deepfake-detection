"""Build a video manifest and extract aligned face crops (Sections 5.1.1 and 5.2.1).

  python prepare_data.py ffpp    --root /content/FF --compression c23 --out data/ffpp_c23
  python prepare_data.py ffpp    --root /content/FF --compression c40 --out data/ffpp_c40
  python prepare_data.py celebdf --root /content/Celeb-DF-v2 --out data/celebdf
  python prepare_data.py dfdc    --root /content/dfdc_test --out data/dfdc

Robustness sets (test split only, perturbation applied before face detection):
  python prepare_data.py ffpp --root /content/FF --compression c23 --split test \
      --perturb jpeg:30 --out data/ffpp_c23_jpeg30

Ablation sets: --detector haar|mtcnn (Table 7.5) and --clip-len 32 (T = 32).

Output: <out>/manifest.csv and <out>/crops/<video_id>/{<frame>.jpg, frames.json}.
Re-running skips videos that are already done, so a Colab session can be resumed.
"""
import argparse
import csv
import glob
import json
import os
from functools import partial

import cv2
import numpy as np

import face_pipeline
from perturb import perturb

FIELDS = ["video_id", "path", "label", "split", "method", "group"]
FF_METHODS = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS", "NeuralTextures": "NT"}


def ffpp_index(root, compression):
    """{(method, file stem): path} for every FF++ video under root. Works for the
    official download (original_sequences/youtube/c23/videos/000.mp4,
    manipulated_sequences/Deepfakes/c23/videos/000_003.mp4) and for one-folder-per-method
    mirrors such as Kaggle's FaceForensics++_C23/{original,Deepfakes,...}/000_003.mp4."""
    index = {}
    for p in glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True):
        parts = os.path.normpath(os.path.relpath(p, root)).split(os.sep)
        if {"raw", "c23", "c40"} & set(parts) and compression not in parts:
            continue                         # official layout holds several compressions
        method = next((short for folder, short in FF_METHODS.items() if folder in parts),
                      "real" if any(x.startswith("original") for x in parts) else None)
        if method:
            index[(method, os.path.splitext(parts[-1])[0])] = p
    return index


def ffpp_rows(root, compression, splits_dir):
    """Official FF++ split: each JSON entry is a [target, source] pair. Both real
    videos and all eight fakes made from the pair share the pair as their group,
    so no identity crosses a split or a cross-validation fold."""
    index = ffpp_index(root, compression)
    rows = []
    for split in ("train", "val", "test"):
        with open(os.path.join(splits_dir, f"{split}.json")) as f:
            pairs = json.load(f)
        for a, b in pairs:
            group = f"{a}_{b}"
            for v in (a, b):
                rows.append(dict(video_id=f"real_{v}", label=0, split=split, method="real", group=group,
                                 path=index.get(("real", v), f"<missing original {v}.mp4>")))
            for short in FF_METHODS.values():
                for t, s in ((a, b), (b, a)):
                    rows.append(dict(video_id=f"{short}_{t}_{s}", label=1, split=split, method=short,
                                     group=group, path=index.get((short, f"{t}_{s}"), f"<missing {short} {t}_{s}.mp4>")))
    return rows


def celebdf_rows(root):
    """Official Celeb-DF v2 test list; in that file 1 = real and 0 = fake."""
    rows = []
    with open(os.path.join(root, "List_of_testing_videos.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            lab, rel = line.split()
            real = lab == "1"
            rows.append(dict(video_id=os.path.splitext(rel.replace("/", "_"))[0], label=int(not real),
                             split="test", method="real" if real else "celebdf", group=rel,
                             path=os.path.join(root, rel)))
    return rows


def dfdc_rows(root):
    """DFDC public test set (labels.csv: filename,label with 1 = fake) or Kaggle
    competition parts (metadata.json per folder). Videos are found anywhere under root."""
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True)}
    labels = {}
    for p in glob.glob(os.path.join(root, "**", "labels.csv"), recursive=True):
        with open(p) as f:
            labels.update({r["filename"]: int(float(r["label"])) for r in csv.DictReader(f)})
    for p in glob.glob(os.path.join(root, "**", "metadata.json"), recursive=True):
        with open(p) as f:
            labels.update({k: int(v["label"] == "FAKE") for k, v in json.load(f).items()})
    return [dict(video_id=os.path.splitext(name)[0], label=lab, split="test",
                 method="real" if lab == 0 else "dfdc", group=name, path=paths[name])
            for name, lab in sorted(labels.items()) if name in paths]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["ffpp", "celebdf", "dfdc"])
    ap.add_argument("--root", required=True, help="dataset root folder")
    ap.add_argument("--out", required=True, help="output folder, e.g. data/ffpp_c23")
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    ap.add_argument("--splits-dir", help="folder with FF++ train/val/test.json (default: <root>/splits)")
    ap.add_argument("--split", choices=["train", "val", "test"], help="only this split")
    ap.add_argument("--perturb", help="kind:level, e.g. jpeg:30, resize:0.25, noise:20, blur:3")
    ap.add_argument("--detector", default="retinaface", choices=sorted(face_pipeline.DETECTORS))
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--n-uniform", type=int, default=32, help="uniform frames per video")
    ap.add_argument("--n-clips", type=int, default=8, help="clips of --clip-len consecutive frames")
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run the detector on every k-th clip frame, interpolate the rest (quick mode: 4)")
    ap.add_argument("--max-side", type=int, default=640, help="detection resolution")
    ap.add_argument("--subsample", type=int, default=1, help="keep every k-th video (quick robustness sets)")
    ap.add_argument("--shard", default="0/1", help="i/n: extract every n-th video starting at i "
                    "(run one process per GPU, e.g. 0/2 and 1/2, into the same --out)")
    a = ap.parse_args()
    shard, n_shards = map(int, a.shard.split("/"))

    if a.dataset == "ffpp":
        rows = ffpp_rows(a.root, a.compression, a.splits_dir or os.path.join(a.root, "splits"))
    else:
        rows = celebdf_rows(a.root) if a.dataset == "celebdf" else dfdc_rows(a.root)
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    rows = rows[::a.subsample]
    os.makedirs(a.out, exist_ok=True)
    if shard == 0:                           # one writer for the shared manifest
        with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"[INFO] {len(rows)} videos in manifest ({sum(r['label'] for r in rows)} fake)")

    fn = None
    if a.perturb:
        kind, level = a.perturb.split(":")
        fn = partial(perturb, kind=kind, level=float(level), rng=np.random.default_rng(0))

    flagged = 0
    for k, r in enumerate(rows, 1):
        if (k - 1) % n_shards != shard:
            continue
        vdir = os.path.join(a.out, "crops", r["video_id"])
        if os.path.exists(os.path.join(vdir, "frames.json")):
            continue
        if not os.path.exists(r["path"]):
            print(f"  [MISSING] {r['path']}")
            continue
        try:
            res = face_pipeline.extract(r["path"], detector=a.detector, perturb_fn=fn, clip_len=a.clip_len,
                                        n_uniform=a.n_uniform, n_clips=a.n_clips, max_side=a.max_side,
                                        detect_every=a.detect_every)
        except Exception as e:                  # corrupt file: report and move on
            print(f"  [ERROR] {r['video_id']}: {e}")
            continue
        os.makedirs(vdir, exist_ok=True)
        for i, crop in res.pop("crops").items():
            cv2.imwrite(os.path.join(vdir, f"{i}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(vdir, "frames.json"), "w") as f:   # written last: marks the video done
            json.dump(res, f)
        flagged += res["flagged"]
        print(f"  [{k}/{len(rows)}] {r['video_id']}: face in {res['n_found']}/{res['n_sampled']} frames"
              + ("  FLAGGED (no face in > half)" if res["flagged"] else ""))
    print(f"[DONE] {flagged} newly flagged videos (excluded from training, counted in evaluate.py)")


if __name__ == "__main__":
    main()
