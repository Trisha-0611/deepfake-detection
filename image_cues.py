"""Spatial and frequency cues (Sections 5.1.2 and 5.1.3).

spatial   : EfficientNet-B0 on RGB face crops
frequency : MobileNetV2 on the centred log-magnitude FFT spectrum (Eq. 5.1, 5.2)

Both use GAP -> Dense(256, ReLU) -> Dropout(0.5) -> sigmoid and a two-stage
schedule: head with the backbone frozen, then the top 20 backbone layers at a
ten times lower learning rate. Augmentation (flip, colour jitter, blur sigma <= 0.5,
JPEG quality 60-100) is applied to the RGB crop, before the FFT.
Trained through train.py; a video's score is the mean over its uniform frames.
"""
import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import data

IMG = 224
FINETUNE_LAYERS = 20


def spectrum(x):
    """(B, H, W, 3) RGB in [0, 255] -> per-channel log(1 + |shift(FFT2(x))|) rescaled to [0, 255]."""
    x = tf.transpose(tf.cast(x, tf.float32), [0, 3, 1, 2])            # FFT over H, W
    f = tf.signal.fftshift(tf.signal.fft2d(tf.complex(x, tf.zeros_like(x))), axes=(-2, -1))
    m = tf.math.log1p(tf.abs(f))
    lo = tf.reduce_min(m, axis=(-2, -1), keepdims=True)
    hi = tf.reduce_max(m, axis=(-2, -1), keepdims=True)
    return tf.transpose(tf.round(255.0 * (m - lo) / (hi - lo + 1e-8)), [0, 2, 3, 1])


def _blur(img):
    """Gaussian blur with sigma ~ U(0, 0.5], 5-tap kernel, reflect padding."""
    s = tf.random.uniform([], 0.05, 0.5)
    k = tf.exp(-tf.range(-2.0, 3.0) ** 2 / (2 * s * s))
    k = k / tf.reduce_sum(k)
    kernel = tf.tile((k[:, None] * k[None, :])[:, :, None, None], [1, 1, 3, 1])
    padded = tf.pad(img[None], [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
    return tf.nn.depthwise_conv2d(padded, kernel, [1, 1, 1, 1], "VALID")[0]


def augment(img):
    """uint8 RGB -> augmented float32 RGB in [0, 255]."""
    x = tf.image.convert_image_dtype(img, tf.float32)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    x = tf.image.random_saturation(x, 0.9, 1.1)
    x = tf.clip_by_value(_blur(tf.clip_by_value(x, 0.0, 1.0)), 0.0, 1.0)
    x = tf.image.random_jpeg_quality(x, 60, 100)
    return x * 255.0


def frame_items(videos, n_frames=None):
    """Paths, labels and video index of the uniform frames (optionally n_frames per video)."""
    paths, labels, idx = [], [], []
    for i, v in enumerate(videos):
        frames = v["uniform"]
        if n_frames and len(frames) > n_frames:
            frames = [frames[j] for j in np.linspace(0, len(frames) - 1, n_frames).round().astype(int)]
        for f in frames:
            paths.append(os.path.join(v["dir"], f"{f}.jpg"))
            labels.append(float(v["label"]))
            idx.append(i)
    return paths, np.array(labels, np.float32), np.array(idx)


def dataset(videos, cue, train, batch=32, n_frames=None, weights=None, seed=0):
    paths, labels, idx = frame_items(videos, n_frames)
    w = np.array([weights[int(y)] for y in labels], np.float32) if weights else np.ones_like(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, w))
    if train:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(p, y, w):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = augment(img) if train else tf.cast(img, tf.float32)
        img.set_shape((IMG, IMG, 3))
        if cue == "frequency":
            img = spectrum(img[None])[0]
        return img, y, w

    return ds.map(load, tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE), idx


def build(cue):
    inp = keras.Input((IMG, IMG, 3), name="spectrum" if cue == "frequency" else "rgb")
    if cue == "spatial":                     # EfficientNet rescales internally
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
        x = inp
    else:
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
        x = layers.Rescaling(1 / 127.5, offset=-1)(inp)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name=f"{cue}_cue"), base


def train(cue, train_v, val_v, out, seed=0, train_frames=None, epochs=(10, 10)):
    keras.utils.set_random_seed(seed)
    model, base = build(cue)
    tr, _ = dataset(train_v, cue, True, n_frames=train_frames, weights=data.class_weights(train_v), seed=seed)
    va, _ = dataset(val_v, cue, False, n_frames=8)           # 8 frames per video keeps validation fast
    stop = lambda: [keras.callbacks.EarlyStopping("val_auc", mode="max", patience=3, restore_best_weights=True),
                    keras.callbacks.CSVLogger(os.path.join(out, "log.csv"), append=True)]
    compile_ = lambda lr: model.compile(keras.optimizers.Adam(lr), "binary_crossentropy",
                                        metrics=[keras.metrics.AUC(name="auc")])
    compile_(1e-3)                           # stage 1: head only
    model.fit(tr, validation_data=va, epochs=epochs[0], callbacks=stop())
    base.trainable = True                    # stage 2: top layers, BatchNorm stays frozen
    for layer in base.layers:
        layer.trainable = layer in base.layers[-FINETUNE_LAYERS:] and not isinstance(layer, layers.BatchNormalization)
    compile_(1e-4)
    model.fit(tr, validation_data=va, epochs=epochs[1], callbacks=stop())
    model.save(os.path.join(out, "model.keras"))
    if cue == "frequency":                   # reference for spectral band occlusion (Section 5.1.6)
        reals = [v for v in train_v if v["label"] == 0][:500]
        ds, _ = dataset(reals, cue, False, n_frames=4)
        np.save(os.path.join(out, "mean_real_spectrum.npy"),
                np.mean(np.concatenate([x.numpy() for x, _, _ in ds]), axis=0))


def load(cue, out):
    return keras.models.load_model(os.path.join(out, "model.keras"))


def predict(cue, model, videos):
    """Video-level fake probability: mean over the video's uniform frames."""
    videos = [v for v in videos if v["uniform"]]
    ds, idx = dataset(videos, cue, False, batch=64)
    p = model.predict(ds.map(lambda x, y, w: x), verbose=0)[:, 0]
    return videos, np.bincount(idx, p, len(videos)) / np.bincount(idx, minlength=len(videos))
