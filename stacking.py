"""Out-of-fold stacking, calibration and fusion baselines (Section 5.1.5, Appendix A.3).

Reads the branch predictions written by train.py and fuses them.

  python stacking.py --run runs/ffpp_c23 --eval test celebdf dfdc ffpp_c23_jpeg30 ...
  python stacking.py --run runs/ffpp_c23 --cues spatial,frequency      # cue subsets (Table 7.4)
  python stacking.py --run runs/ffpp_c23 --fusion mean|wmean|logreg     # simpler rules
  python stacking.py --run runs/ffpp_c23 --meta-source insample         # "Stacking, no OOF"
  python stacking.py --run runs/ffpp_c23 --export models                # models for detector.py / app.py

--meta-source val trains the meta-learner on validation predictions instead of
out-of-fold ones (cheap option for the leave-one-method-out runs; say so if used).
Output: <run>/fusion/<tag>/seed<seed>/pred_<set>.csv (+ meta.keras, calib.json).
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import keras
from keras import layers
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from explain import CUES, conflict_band, conflict_index, shapley_values


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_meta_learner(n_inputs):
    inp = layers.Input(shape=(n_inputs,))                 # cue logits (+ temporal mask)
    h = layers.Dense(32, activation="relu")(inp)
    h = layers.Dense(16, activation="relu")(h)
    out = layers.Dense(1)(h)                              # fused logit g(z)
    model = keras.Model(inp, out, name="meta_learner")
    model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))
    return model


def fit_temperature(logits, y):
    """Single temperature minimising validation negative log-likelihood."""
    def nll(t):
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def features(x, cues, mask):
    """z = [logit of each cue, temporal availability flag]. x: {cue: logit array}."""
    cols = [x[c] for c in cues] + ([mask] if "temporal" in cues else [])
    return np.column_stack([np.broadcast_to(c, np.shape(mask)) for c in cols]).astype(np.float32)


def fused_logit(meta, x, mask, cues=CUES):
    return np.asarray(meta(features(x, cues, mask), training=False))[:, 0]


def cue_table(run, cues, seed, name):
    """One row per video: label, method and each cue's probability (NaN if missing)."""
    frames = {}
    for c in cues:
        d = os.path.join(run, c, f"seed{seed}")
        files = ([os.path.join(d, f"fold{k}", "pred_oof.csv") for k in range(5)] if name == "oof"
                 else [os.path.join(d, "full", f"pred_{name}.csv")])
        frames[c] = pd.concat([pd.read_csv(f) for f in files]).set_index("video_id")
    df = pd.concat([f[["label", "method"]] for f in frames.values()]).groupby(level=0).first()
    for c in cues:
        df[c] = frames[c]["prob"]
    return df.dropna(subset=[c for c in cues if c != "temporal"])


def logits_and_mask(df, cues):
    x = {c: logit(df[c].fillna(0.5).to_numpy()) for c in cues}          # missing p_t = 0.5 -> logit 0
    mask = df["temporal"].notna().to_numpy(np.float32) if "temporal" in df else np.ones(len(df), np.float32)
    return x, mask


def tag_of(a, cues):
    tag = a.fusion
    if len(cues) < 3:
        tag += "_" + "".join(c[0].upper() for c in cues)
    return tag + {"oof": "", "insample": "_nooof", "val": "_valmeta"}[a.meta_source]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. runs/ffpp_c23")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cues", default=",".join(CUES))
    ap.add_argument("--fusion", default="stacking", choices=["stacking", "mean", "wmean", "logreg"])
    ap.add_argument("--meta-source", default="oof", choices=["oof", "insample", "val"])
    ap.add_argument("--eval", nargs="*", default=["test"], help="prediction sets to fuse")
    ap.add_argument("--mask-rate", type=float, default=0.2)
    ap.add_argument("--export", help="copy the deployable models here (stacking, all cues)")
    a = ap.parse_args()
    cues = [c for c in CUES if c in a.cues.split(",")]
    keras.utils.set_random_seed(a.seed)

    tr = cue_table(a.run, cues, a.seed, {"oof": "oof", "insample": "train", "val": "val"}[a.meta_source])
    va = cue_table(a.run, cues, a.seed, "val")
    x_tr, m_tr = logits_and_mask(tr, cues)
    x_va, m_va = logits_and_mask(va, cues)
    y_tr, y_va = tr.label.to_numpy(np.float32), va.label.to_numpy(np.float32)

    out = os.path.join(a.run, "fusion", tag_of(a, cues), f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)

    if a.fusion == "stacking":
        av_tr, av_va = tr[cues].notna().to_numpy(), va[cues].notna().to_numpy()
        baseline = {c: float(np.mean(x_tr[c][(y_tr == 0) & av_tr[:, j]])) for j, c in enumerate(cues)}
        if "temporal" in cues:               # hide the temporal cue in 20 % of rows (still images)
            hide = np.random.default_rng(a.seed).random(len(tr)) < a.mask_rate
            x_tr["temporal"] = np.where(hide, 0.0, x_tr["temporal"])
            m_tr = np.where(hide, 0.0, m_tr).astype(np.float32)
        meta = build_meta_learner(len(cues) + ("temporal" in cues))
        meta.fit(features(x_tr, cues, m_tr), y_tr, validation_data=(features(x_va, cues, m_va), y_va),
                 epochs=200, batch_size=256, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=20, restore_best_weights=True)])
        g = lambda x, m: fused_logit(meta, x, m, cues)
        cue_t = {c: fit_temperature(x_va[c][av_va[:, j]], y_va[av_va[:, j]]) for j, c in enumerate(cues)}
        k_va = [conflict_index([sigmoid(x_va[c][i] / cue_t[c]) for j, c in enumerate(cues) if row[j]])
                for i, row in enumerate(av_va)]
        calib = {"cues": cues, "T_c": fit_temperature(g(x_va, m_va), y_va), "cue_T": cue_t,
                 "baseline": baseline, "conflict_terciles": np.quantile(k_va, [1 / 3, 2 / 3]).tolist(),
                 "prior": float(y_va.mean())}      # fake share the fused probability is calibrated to
        meta.save(os.path.join(out, "meta.keras"))
        with open(os.path.join(out, "calib.json"), "w") as f:
            json.dump(calib, f, indent=2)
    elif a.fusion == "logreg":
        lr = LogisticRegression(max_iter=1000).fit(features(x_tr, cues, m_tr), y_tr)
    elif a.fusion == "wmean":
        w = {c: roc_auc_score(va.label[va[c].notna()], va[c].dropna()) for c in cues}

    for name in a.eval:
        df = cue_table(a.run, cues, a.seed, name)
        x, m = logits_and_mask(df, cues)
        res = df[["label", "method"]].copy()
        if a.fusion == "stacking":
            gz = g(x, m)
            res["prob"] = sigmoid(gz / calib["T_c"])
            base = {c: np.full(len(df), calib["baseline"][c]) for c in cues}
            unavail = np.column_stack([(m == 0) if c == "temporal" else np.zeros(len(df), bool) for c in cues])
            phi = {c: np.zeros(len(df)) for c in cues}
            # Shapley over the cues each video actually has (no temporal clip -> two-cue game)
            groups = ([(m == 1, cues), (m == 0, [c for c in cues if c != "temporal"])] if "temporal" in cues
                      else [(np.ones(len(df), bool), cues)])
            for rows, av in groups:
                if not rows.any():
                    continue
                sub = {c: x[c][rows] for c in cues}
                part = shapley_values(lambda xx: g({**sub, **xx}, m[rows]), sub,
                                      {c: base[c][rows] for c in cues}, available=av)
                for c in av:
                    phi[c][rows] = part[c]
            for c in cues:
                res[f"phi_{c}"] = phi[c]
            res["conflict"] = [conflict_index([sigmoid(x[c][i] / calib["cue_T"][c])
                                               for j, c in enumerate(cues) if not unavail[i, j]])
                               for i in range(len(df))]
            res["band"] = [conflict_band(k, *calib["conflict_terciles"]) for k in res["conflict"]]
            # cue-removal fidelity: set the top / bottom |phi| cue to its baseline
            P = np.abs(np.column_stack([phi[c] for c in cues]))
            top, bottom = np.where(unavail, -np.inf, P).argmax(1), np.where(unavail, np.inf, P).argmin(1)
            for col, j_of in (("d_top", top), ("d_bottom", bottom)):
                xr = {c: np.where(j_of == j, base[c], x[c]) for j, c in enumerate(cues)}
                res[col] = np.abs(res["prob"] - sigmoid(g(xr, m) / calib["T_c"]))
        elif a.fusion == "logreg":
            res["prob"] = lr.predict_proba(features(x, cues, m))[:, 1]
        else:
            probs = df[cues].to_numpy()
            wts = np.ones(len(cues)) if a.fusion == "mean" else np.array([w[c] for c in cues])
            ok = ~np.isnan(probs)
            res["prob"] = np.nansum(probs * wts, 1) / (ok * wts).sum(1)
        res.to_csv(os.path.join(out, f"pred_{name}.csv"))
        print(f"[OK] {out}/pred_{name}.csv  AUC {roc_auc_score(res.label, res.prob):.4f}")

    if a.export:
        assert a.fusion == "stacking" and len(cues) == 3, "export the proposed model (stacking, all cues)"
        os.makedirs(a.export, exist_ok=True)
        for c in cues:
            shutil.copy(os.path.join(a.run, c, f"seed{a.seed}", "full", "model.keras"),
                        os.path.join(a.export, f"{c}.keras"))
        shutil.copy(os.path.join(a.run, "frequency", f"seed{a.seed}", "full", "mean_real_spectrum.npy"), a.export)
        for f in ("meta.keras", "calib.json"):
            shutil.copy(os.path.join(out, f), a.export)
        print(f"[OK] exported models to {a.export}/")


if __name__ == "__main__":
    main()
