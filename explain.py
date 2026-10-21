"""Explanation module (Section 5.1.6, Appendix A.4).

Cue level   : exact Shapley values of the fused logit over the cues, and the
              normalised cue-conflict index.
Feature level: Grad-CAM (spatial), radial band occlusion (frequency); frame
              occlusion (temporal) lives in temporal.frame_importance.
Faithfulness: Grad-CAM deletion AUC against random orderings.
"""
from itertools import combinations
from math import factorial

import numpy as np

CUES = ("spatial", "frequency", "temporal")


def shapley_values(g, z, baseline, available=CUES):
    """Exact Shapley values of the fusion function g over the available cues.

    g        : callable mapping a dict {cue: logit} to a fused logit
               (works element-wise when the logits are arrays)
    z        : dict {cue: logit} for the input being explained
    baseline : dict {cue: mean logit of that cue on real reference videos (here the validation split)}
    Returns  : dict {cue: phi}; sum(phi) == g(z) - g(baseline).
    """
    n = len(available)
    phi = {c: 0.0 for c in available}

    def value(coalition):
        x = {c: (z[c] if c in coalition else baseline[c]) for c in available}
        return g(x)

    cache = {}
    for r in range(n + 1):
        for s in combinations(available, r):
            cache[frozenset(s)] = value(set(s))
    for i in available:
        others = [c for c in available if c != i]
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for s in combinations(others, r):
                s = frozenset(s)
                phi[i] = phi[i] + w * (cache[s | {i}] - cache[s])
    return phi


def conflict_index(probs):
    """Variance of N calibrated cue probabilities divided by its maximum
    floor(N/2) * ceil(N/2) / N**2, so the index lies in [0, 1] for any N >= 2."""
    p = np.asarray(probs, dtype=float)
    n = p.size
    if n < 2:
        return 0.0
    k_max = (n // 2) * ((n + 1) // 2) / n ** 2
    return float(np.var(p) / k_max)


def conflict_band(k, t_low, t_high):
    """Map the index to low / medium / high using validation-set terciles."""
    return "low" if k < t_low else ("medium" if k < t_high else "high")


def grad_cam(model, img):
    """Grad-CAM of the fake probability w.r.t. the backbone's last feature map.
    img: (224, 224, 3) float32 model input. Returns a (224, 224) map in [0, 1]."""
    import cv2
    import keras
    import tensorflow as tf
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    pos = model.layers.index(base)
    x = tf.constant(img[None], tf.float32)
    for layer in model.layers[1:pos]:
        x = layer(x)
    with tf.GradientTape() as tape:
        fmap = base(x, training=False)
        tape.watch(fmap)
        y = fmap
        for layer in model.layers[pos + 1:]:
            y = layer(y, training=False)
    grads = tape.gradient(y, fmap)[0]
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * tf.reduce_mean(grads, axis=(0, 1)), axis=-1)).numpy()
    cam = cv2.resize(cam, img.shape[1::-1])
    return cam / (cam.max() + 1e-8)


def _radius(size=224):
    yy, xx = np.mgrid[:size, :size]
    return np.hypot(yy - size // 2, xx - size // 2)       # zero frequency sits at size//2 after the shift


def band_importance(model, spec, mean_real_spec, n_bands=8):
    """Drop in p_f when each of n_bands concentric radial bands of the spectrum
    is replaced by the mean spectrum of real training faces."""
    r = _radius(spec.shape[0])
    edges = np.linspace(0, r.max() + 1e-6, n_bands + 1)
    batch = np.repeat(spec[None], n_bands + 1, axis=0)
    for b in range(n_bands):
        mask = (r >= edges[b]) & (r < edges[b + 1])
        batch[b + 1][mask] = mean_real_spec[mask]
    p = model.predict(batch, verbose=0)[:, 0]
    return p[0] - p[1:]


def deletion_curve(model, img, order, steps=10):
    """p_s as the pixels in `order` are replaced by a blurred copy in 10 % steps."""
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    flat, flat_blur = img.reshape(-1, 3), blurred.reshape(-1, 3)
    batch = []
    for s in range(steps + 1):
        x = flat.copy()
        k = order[: len(order) * s // steps]
        x[k] = flat_blur[k]
        batch.append(x.reshape(img.shape))
    return model.predict(np.stack(batch), verbose=0)[:, 0]


def deletion_auc(model, img, rng):
    """(Grad-CAM deletion AUC, random-order deletion AUC); lower = more faithful."""
    order = np.argsort(-grad_cam(model, img).ravel())
    auc = lambda c: float(np.mean((c[1:] + c[:-1]) / 2))
    return auc(deletion_curve(model, img, order)), auc(deletion_curve(model, img, rng.permutation(order.size)))


# Rough facial landmarks of the aligned 224 x 224 crop (make_template(224, 1.3)).
_REGIONS = {"the left eye": (85, 104), "the right eye": (139, 104), "the nose": (112, 136),
            "the mouth": (112, 168)}


def cam_region(cam):
    """Name the facial region at the Grad-CAM peak."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    name, (cx, cy) = min(_REGIONS.items(), key=lambda kv: np.hypot(kv[1][0] - x, kv[1][1] - y))
    return name if np.hypot(cx - x, cy - y) < 40 else "the face boundary"


def render_report(r):
    """Fixed template: every statement maps to a measured quantity."""
    phi = ", ".join(f"{c} {r['shapley'][c]:+.2f}" for c in CUES if c in r["shapley"])
    lines = [f"Verdict: {r['verdict']} (confidence {r['confidence']:.2f}). "
             f"Contributions to the fake score (logit units): {phi}. "
             f"The spatial evidence concentrates on {r['cam_region']}. "
             f"Cue conflict: {r['band']} (k = {r['conflict']:.2f})."]
    if "temporal" not in r["probs"]:
        lines.append("The temporal cue was not used because no clip of consecutive face frames was available.")
    if r["band"] == "high":
        lines.append("The cues disagree; treat this verdict with caution.")
    lines.append("This result is evidence for a human decision, not proof.")
    return " ".join(lines)
