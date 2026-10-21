"""Inference on one video or image (Section 4.2.4, Fig. 4.6).

  python detector.py --input video.mp4 [--models models] [--detector retinaface]

Loads the models exported by  stacking.py --export models : spatial.keras,
frequency.keras, temporal.keras, meta.keras, calib.json, mean_real_spectrum.npy.
"""
import argparse
import json
import os
import time

import face_pipeline                        # first: it loads RetinaFace before Keras 3
import cv2
import numpy as np
import keras

import explain
import temporal
from explain import CUES
from image_cues import spectrum
from stacking import fused_logit, logit, sigmoid

MODEL_FILES = [f"{c}.keras" for c in CUES] + ["meta.keras", "calib.json", "mean_real_spectrum.npy"]


class DeepfakeDetector:
    def __init__(self, models_dir="models", threshold=0.5, detector="retinaface", max_side=320):
        missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(models_dir, f))]
        if missing:
            raise FileNotFoundError(f"missing in {models_dir}/: {', '.join(missing)} "
                                    "(train the models, then run stacking.py --export models)")
        load = lambda n: keras.models.load_model(os.path.join(models_dir, n))
        self.models = {c: load(f"{c}.keras") for c in CUES}
        self.meta = load("meta.keras")
        with open(os.path.join(models_dir, "calib.json")) as f:
            self.calib = json.load(f)
        self.mean_spec = np.load(os.path.join(models_dir, "mean_real_spectrum.npy"))
        self.extractor = temporal.feature_extractor()
        # detection on a 320 px copy keeps CPU analysis of a 10 s video near NFR1's 60 s;
        # crops are still cut from the full-resolution frame
        self.threshold, self.detector, self.max_side = threshold, detector, max_side

    def detect(self, path, progress=lambda stage: None):
        """progress(stage) is called as each stage starts (the app shows these)."""
        t0 = time.perf_counter()
        progress("Detecting, tracking and aligning the face")
        try:
            fp = face_pipeline.extract(path, detector=self.detector, max_side=self.max_side)
        except Exception as e:
            return {"error": f"Could not read the file: {e}"}
        if fp["n_sampled"] == 0:
            return {"error": "Could not decode the file. It may be corrupt or use an unsupported codec."}
        if fp["flagged"]:
            return {"error": "No face was found in more than half of the analysed frames; no verdict is issued.",
                    "flagged": True, "n_sampled": fp["n_sampled"], "n_found": fp["n_found"]}
        crops = {i: cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for i, c in fp["crops"].items()}
        t1 = time.perf_counter()
        progress(f"Running the spatial, frequency and temporal cues on {fp['n_found']} face crops")

        frames = np.stack([crops[i] for i in fp["uniform"]]).astype(np.float32)
        specs = spectrum(frames).numpy()
        p_s = self.models["spatial"].predict(frames, verbose=0)[:, 0]
        p_f = self.models["frequency"].predict(specs, verbose=0)[:, 0]
        probs = {"spatial": float(p_s.mean()), "frequency": float(p_f.mean())}
        feats = clip_p = None
        if fp["clips"]:
            feats = temporal.clip_features(self.extractor, np.stack([[crops[i] for i in c] for c in fp["clips"]]))
            clip_p = self.models["temporal"].predict(feats, verbose=0)[:, 0]
            probs["temporal"] = float(clip_p.mean())
        t2 = time.perf_counter()
        progress("Fusing the cues and computing the explanations")

        mask = np.array([1.0 if "temporal" in probs else 0.0])
        z = {c: np.array([logit(probs.get(c, 0.5))]) for c in CUES}
        g = lambda x: fused_logit(self.meta, {**z, **x}, mask)
        # The fused probability is calibrated on FF++ validation, where ~80% of videos are fake. Remove
        # that prior so 0.5 means "equally likely real or fake" for an unknown video.
        p = float(sigmoid(g(z)[0] / self.calib["T_c"] - logit(self.calib.get("prior", 0.5))))
        phi = explain.shapley_values(g, z, {c: np.array([v]) for c, v in self.calib["baseline"].items()},
                                     available=[c for c in CUES if c in probs])
        cal = [sigmoid(logit(probs[c]) / self.calib["cue_T"][c]) for c in CUES if c in probs]
        k = explain.conflict_index(cal)

        top = int(np.argmax(p_s))
        cam = explain.grad_cam(self.models["spatial"], frames[top])
        heat = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        sel = np.argsort(-p_f)[:4]
        bands = np.mean([explain.band_importance(self.models["frequency"], specs[i], self.mean_spec) for i in sel], axis=0)
        r = {
            "verdict": "FAKE" if p >= self.threshold else "REAL",
            "confidence": max(p, 1 - p),
            "fusion_prob": p,
            "probs": probs,
            "shapley": {c: float(v[0]) for c, v in phi.items()},
            "conflict": k,
            "band": explain.conflict_band(k, *self.calib["conflict_terciles"]),
            "face": frames[top].astype(np.uint8),
            "gradcam": (0.55 * frames[top] + 0.45 * heat).astype(np.uint8),
            "cam_region": explain.cam_region(cam),
            "band_scores": bands.tolist(),
            "n_frames": fp["n_frames"], "n_sampled": fp["n_sampled"], "n_found": fp["n_found"],
        }
        if feats is not None:
            j = int(np.argmax(clip_p))
            r["frame_importance"] = temporal.frame_importance(self.models["temporal"], feats[j]).tolist()
            r["frame_indices"] = fp["clips"][j]
        r["report"] = explain.render_report(r)
        t3 = time.perf_counter()
        r["timing"] = {"preprocessing": t1 - t0, "cues": t2 - t1, "fusion_explanation": t3 - t2}
        return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--detector", default="retinaface", choices=sorted(face_pipeline.DETECTORS))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-side", type=int, default=320, help="detection resolution")
    a = ap.parse_args()
    r = DeepfakeDetector(a.models, a.threshold, a.detector, a.max_side).detect(a.input)
    if "error" in r:
        raise SystemExit(f"[ERROR] {r['error']}")
    print("\n" + r["report"] + "\n")
    print("  cue probabilities :", {c: round(v, 4) for c, v in r["probs"].items()})
    print("  Shapley (logit)   :", {c: round(v, 4) for c, v in r["shapley"].items()})
    print(f"  fused p = {r['fusion_prob']:.4f}   conflict k = {r['conflict']:.3f} ({r['band']})")
    print(f"  faces found in {r['n_found']}/{r['n_sampled']} sampled frames")
    print("  timing (s)        :", {k: round(v, 2) for k, v in r["timing"].items()},
          f"total {sum(r['timing'].values()):.1f}")


if __name__ == "__main__":
    main()
