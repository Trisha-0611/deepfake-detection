"""Temporal cue (Section 5.1.4, Appendix A.2): per-frame appearance and
frame-difference features from a frozen EfficientNet-B0, then a BiLSTM with
attention pooling over clips of T consecutive frames.

Features are precomputed once per video (8 clips x T frames x 2560):
  python temporal.py --features data/ffpp_c23 data/ffpp_c40 data/celebdf ...
Training goes through train.py. Variants for Table 7.5:
  temporal (T = 16), temporal_T8, temporal_T32 (needs a --clip-len 32 data set),
  temporal_nodiff (appearance stream only).
"""
import argparse
import os

import cv2
import numpy as np
import keras
from keras import layers

import data

FEAT = 1280                  # EfficientNet-B0 pooled feature size


def variant(cue):
    """cue name -> (clip length T, use frame-difference stream)."""
    return {"temporal": (16, True), "temporal_T8": (8, True), "temporal_T32": (32, True),
            "temporal_nodiff": (16, False)}[cue]


def feature_extractor(weights="imagenet"):
    base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                             pooling="avg", input_shape=(224, 224, 3))
    base.trainable = False
    return base


def clip_features(extractor, clips_rgb):
    """clips_rgb: (n_clips, T, 224, 224, 3) uint8 aligned crops -> (n_clips, T, 2 * FEAT)."""
    x = clips_rgb.astype(np.float32)
    diff = np.abs(np.diff(x, axis=1, prepend=x[:, :1]))      # |x_t - x_(t-1)|, zero for t = 1
    n, t = x.shape[:2]
    a = extractor.predict(x.reshape(-1, 224, 224, 3), batch_size=64, verbose=0)
    d = extractor.predict(diff.reshape(-1, 224, 224, 3), batch_size=64, verbose=0)
    return np.concatenate([a, d], axis=-1).reshape(n, t, -1)


def feat_path(v):
    return os.path.join(v["set_dir"], "feats", v["video_id"] + ".npy")


def extract_features(set_dir):
    ext = feature_extractor()
    os.makedirs(os.path.join(set_dir, "feats"), exist_ok=True)
    videos = [v for v in data.load_videos(set_dir) if v["clips"] and not os.path.exists(feat_path(v))]
    for k, v in enumerate(videos, 1):
        clips = np.stack([[cv2.cvtColor(cv2.imread(os.path.join(v["dir"], f"{i}.jpg")), cv2.COLOR_BGR2RGB)
                           for i in c] for c in v["clips"]])
        np.save(feat_path(v), clip_features(ext, clips).astype(np.float16))
        print(f"  [{k}/{len(videos)}] {v['video_id']}")


def build_temporal_model(t=16, feat=2 * FEAT):
    inp = layers.Input(shape=(t, feat))
    h = layers.Dense(256, activation="relu")(inp)
    h = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(h)
    score = layers.Dense(1, activation="tanh")(h)                  # attention pooling over time
    weights = layers.Softmax(axis=1, name="frame_attention")(score)
    pooled = layers.Flatten()(layers.Dot(axes=1)([weights, h]))    # (batch, 256)
    h = layers.Dense(64, activation="relu")(pooled)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model(inp, out, name="temporal_cue")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def _clips(v, t, diff):
    f = np.load(feat_path(v))                # float16, ~0.6 MB per video
    if f.shape[1] < t:
        raise ValueError(f"stored clips have {f.shape[1]} frames, T = {t} needs prepare_data.py --clip-len {t}")
    return f[:, :t] if diff else f[:, :t, :FEAT]


class Clips(keras.utils.PyDataset):
    """Training: a random stored clip per video, redrawn every epoch. Validation: the middle clip."""

    def __init__(self, videos, t, diff, train, weights=None, batch=16, seed=0):
        super().__init__()
        self.feats = [_clips(v, t, diff) for v in videos]
        self.y = np.array([v["label"] for v in videos], np.float32)
        self.w = np.array([weights[int(y)] for y in self.y], np.float32) if weights else np.ones_like(self.y)
        self.train, self.batch, self.rng = train, batch, np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return -(-len(self.y) // self.batch)

    def on_epoch_end(self):
        n = len(self.y)
        self.order = self.rng.permutation(n) if self.train else np.arange(n)
        self.pick = [self.rng.integers(len(f)) if self.train else len(f) // 2 for f in self.feats]

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        x = np.stack([self.feats[j][self.pick[j]] for j in idx]).astype(np.float32)
        return x, self.y[idx], self.w[idx]


def train(cue, train_v, val_v, out, seed=0, **_):
    keras.utils.set_random_seed(seed)
    t, diff = variant(cue)
    train_v = [v for v in train_v if os.path.exists(feat_path(v))]
    val_v = [v for v in val_v if os.path.exists(feat_path(v))]
    if not train_v:
        raise SystemExit("No temporal features found: run  python temporal.py --features <data set>  first.")
    model = build_temporal_model(t, 2 * FEAT if diff else FEAT)
    model.fit(Clips(train_v, t, diff, True, data.class_weights(train_v), seed=seed),
              validation_data=Clips(val_v, t, diff, False), epochs=30,
              callbacks=[keras.callbacks.EarlyStopping("val_auc", mode="max", patience=5, restore_best_weights=True),
                         keras.callbacks.CSVLogger(os.path.join(out, "log.csv"))])
    model.save(os.path.join(out, "model.keras"))


def load(cue, out):
    return keras.models.load_model(os.path.join(out, "model.keras"))


def predict(cue, model, videos):
    """Video-level fake probability: mean over the video's clips. Videos without a
    complete clip get no temporal score (the meta-learner then masks the cue)."""
    t, diff = variant(cue)
    videos = [v for v in videos if os.path.exists(feat_path(v))]
    probs = [float(np.mean(model.predict_on_batch(np.asarray(_clips(v, t, diff), np.float32))))
             for v in videos]
    return videos, np.array(probs)


def frame_importance(model, feats):
    """Frame-occlusion attribution: replace one frame's features with the
    clip mean and record the change in clip probability."""
    base = float(model.predict(feats[None], verbose=0)[0, 0])
    mean = feats.mean(axis=0)
    batch = np.repeat(feats[None], len(feats), axis=0)
    for t in range(len(feats)):
        batch[t, t] = mean
    return base - model.predict(batch, verbose=0)[:, 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="data set folders to extract features for")
    for s in ap.parse_args().features:
        extract_features(s)
