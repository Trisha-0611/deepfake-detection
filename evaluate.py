"""Results tables of Chapter 7 from the prediction files of train.py and stacking.py.

  python evaluate.py                         # -> results/tables.md, results/risk_coverage.png
  python evaluate.py --seeds 0 1 2 --deletion 200

Cells whose runs do not exist yet are shown as "–", so rerun it as experiments
finish. Values are percentages, mean ± std over the seeds found.

Expected run names (see README): runs/ffpp_c23, runs/ffpp_c40, runs/ffpp_c23_no<M>,
runs/ffpp_c23_haar, runs/ffpp_c23_mtcnn, runs/ffpp_c23_clip32; robustness sets
ffpp_c23_<kind><level>, e.g. ffpp_c23_jpeg30, ffpp_c23_resize0.25.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from delong import delong_test
from perturb import LEVELS

METHODS = ["DF", "F2F", "FS", "NT"]
CUES = ["spatial", "frequency", "temporal"]
NAMES = {"spatial": "Spatial cue", "frequency": "Frequency cue", "temporal": "Temporal cue",
         "stacking": "Fused (stacking)"}
STRONGEST = [f"{k}{v[-1]}" for k, v in LEVELS.items()]          # jpeg30, resize0.25, noise20, blur3


def auc(df):
    return roc_auc_score(df.label, df.prob)


def ece(df, bins=15):
    conf = np.maximum(df.prob, 1 - df.prob).to_numpy()
    correct = ((df.prob >= 0.5).astype(int) == df.label).to_numpy()
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


class Results:
    def __init__(self, runs, seeds):
        self.runs, self.seeds = runs, seeds

    def load(self, train, model, seed, evalset, methods=None):
        sub = (os.path.join(model, f"seed{seed}", "full") if model.split("_")[0] in CUES
               else os.path.join("fusion", model, f"seed{seed}"))
        path = os.path.join(self.runs, train, sub, f"pred_{evalset}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        return df[df.method.isin(["real", *methods])] if methods else df

    def values(self, fn, train, model, evalset, methods=None):
        out = []
        for s in self.seeds:
            df = self.load(train, model, s, evalset, methods)
            if df is not None and df.label.nunique() == 2:
                out.append(fn(df))
        return out

    def robust_drop(self, train, model):
        """Clean AUC minus mean AUC over the four perturbations at their strongest level."""
        out = []
        for s in self.seeds:
            clean = self.load(train, model, s, "test")
            pert = [self.load(train, model, s, f"{train}_{p}") for p in STRONGEST]
            pert = [auc(d) for d in pert if d is not None]
            if clean is not None and pert:
                out.append(auc(clean) - np.mean(pert))
        return out

    def delong(self, train, a, b, evalset="test"):
        da, db = self.load(train, a, self.seeds[0], evalset), self.load(train, b, self.seeds[0], evalset)
        if da is None or db is None:
            return "–"
        m = da.merge(db, on="video_id", suffixes=("_a", "_b"))
        return f"{delong_test(m.label_a, m.prob_a, m.prob_b)[2]:.3g}"


def fmt(vals):
    if not len(vals):
        return "–"
    return f"{100 * np.mean(vals):.1f}" + (f" ± {100 * np.std(vals):.1f}" if len(vals) > 1 else "")


def md(title, header, rows, note=""):
    out = f"\n### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    out += "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)
    return out + (f"\n{note}\n" if note else "")


def detection_failure(data_dir, name):
    metas = []
    for p in glob.glob(os.path.join(data_dir, name, "crops", "*", "frames.json")):
        with open(p) as f:
            metas.append(json.load(f))
    if not metas:
        return "–"
    found, sampled = sum(m["n_found"] for m in metas), sum(m["n_sampled"] for m in metas)
    flagged = sum(m["flagged"] for m in metas)
    return f"{100 * (1 - found / max(1, sampled)):.1f} ({flagged} videos flagged)"


def deletion(runs, data_dir, n, out):
    import cv2
    import keras
    import data
    import explain
    model = keras.models.load_model(os.path.join(runs, "ffpp_c23", "spatial", "seed0", "full", "model.keras"))
    rng = np.random.default_rng(0)
    fakes = [v for v in data.load_videos(os.path.join(data_dir, "ffpp_c23"), "test") if v["label"] == 1]
    res = []
    for v in rng.choice(fakes, size=min(n, len(fakes)), replace=False):
        f = v["uniform"][len(v["uniform"]) // 2]
        img = cv2.cvtColor(cv2.imread(os.path.join(v["dir"], f"{f}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32)
        res.append(explain.deletion_auc(model, img, rng))
    res = np.array(res)
    json.dump({"gradcam": float(res[:, 0].mean()), "random": float(res[:, 1].mean()), "n": len(res)},
              open(os.path.join(out, "deletion.json"), "w"))


def risk_coverage(df, out, thr=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = ((df.prob >= thr).astype(int) != df.label).to_numpy()[np.argsort(df.conflict.to_numpy())]
    cov = np.arange(1, len(err) + 1) / len(err)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    plt.figure(figsize=(5, 3.5))
    plt.plot(cov, 100 * risk)
    plt.xlabel("Coverage (fraction of test videos kept)")
    plt.ylabel("Error rate on kept videos (%)")
    plt.xlim(0.5, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "risk_coverage.png"), dpi=200)
    k90 = int(0.9 * len(err))
    return f"Deferring the 10% most-conflicted videos lowers the error from {100 * risk[-1]:.1f}% to {100 * risk[k90 - 1]:.1f}%."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--deletion", type=int, default=0, help="test videos for the Grad-CAM deletion test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fused", default="stacking", help="fusion tag reported as the proposed model "
                    "(stacking_valmeta for the quick run)")
    a = ap.parse_args()
    F = a.fused
    NAMES[F] = NAMES["stacking"]
    os.makedirs(a.out, exist_ok=True)
    R = Results(a.runs, a.seeds)
    models = CUES + [F]
    t = []

    rows = [[NAMES[m]] + [fmt(R.values(auc, "ffpp_c23", m, "test", [x])) for x in METHODS]
            + [fmt(R.values(auc, "ffpp_c23", m, "test")), fmt(R.values(auc, "ffpp_c40", m, "test")),
               fmt(R.values(ece, "ffpp_c23", m, "test"))] for m in models]
    note = "DeLong p, fused vs each cue (c23, seed %d): " % a.seeds[0] + ", ".join(
        f"{c} {R.delong('ffpp_c23', F, c)}" for c in CUES)
    t.append(md("Table 7.1: AUC (%) per method on the FF++ test split, with ECE (%)",
                ["Model", "DF c23", "F2F c23", "FS c23", "NT c23", "All c23", "All c40", "ECE c23"], rows, note))

    acc = lambda d: ((d.prob >= 0.5).astype(int) == d.label).mean()
    f1 = lambda d: f1_score(d.label, d.prob >= 0.5)
    ap_ = lambda d: average_precision_score(d.label, d.prob)
    tnr = lambda d: ((d.prob < 0.5) & (d.label == 0)).sum() / (d.label == 0).sum()
    t.append(md("Supplementary: threshold metrics on FF++ c23 test at tau = 0.5 (%)",
                ["Model", "Accuracy", "F1", "AP", "Real videos kept REAL"],
                [[NAMES[m], fmt(R.values(acc, "ffpp_c23", m, "test")), fmt(R.values(f1, "ffpp_c23", m, "test")),
                  fmt(R.values(ap_, "ffpp_c23", m, "test")), fmt(R.values(tnr, "ffpp_c23", m, "test"))] for m in models]))

    def eer_rates(m, evalset):
        """(fakes caught, reals kept) per seed at the threshold where FPR = FNR on the validation split."""
        tp, tn = [], []
        for s in a.seeds:
            va, ev = R.load("ffpp_c23", m, s, "val"), R.load("ffpp_c23", m, s, evalset)
            if va is None or ev is None:
                continue
            fpr, tpr, th = roc_curve(va.label, va.prob)
            thr = th[np.argmin(np.abs(1 - tpr - fpr))]
            tp.append(((ev.prob >= thr) & (ev.label == 1)).sum() / (ev.label == 1).sum())
            tn.append(((ev.prob < thr) & (ev.label == 0)).sum() / (ev.label == 0).sum())
        return tp, tn
    rows = [[NAMES[m]] + [fmt(v) for e in ("test", "celebdf") for v in eer_rates(m, e)] for m in models]
    t.append(md("Supplementary: detection rates at the validation equal-error threshold (%)",
                ["Model", "FF++ fakes caught", "FF++ reals kept REAL", "Celeb-DF fakes caught", "Celeb-DF reals kept REAL"],
                rows, "The threshold is chosen on the validation split only (false-positive rate = false-negative rate)."))

    rows = []
    for m in models:
        per = [R.values(auc, f"ffpp_c23_no{x}", m, "test", [x]) for x in METHODS]
        rows.append([NAMES[m]] + [fmt(v) for v in per] + [fmt([np.mean(v) for v in per if v])])
    t.append(md("Table 7.2: Leave-one-method-out AUC (%) on FF++ c23", ["Model", "Held out: DF", "F2F", "FS", "NT", "Mean"], rows))

    t.append(md("Table 7.3: Comparison with published methods, AUC (%) - project row",
                ["Method", "Cues", "FF++ c23", "FF++ c40", "Celeb-DF v2", "DFDC"],
                [["This project (fused)", "S + F + T", fmt(R.values(auc, "ffpp_c23", F, "test")),
                  fmt(R.values(auc, "ffpp_c40", F, "test")), fmt(R.values(auc, "ffpp_c23", F, "celebdf")),
                  fmt(R.values(auc, "ffpp_c23", F, "dfdc"))]],
                "Copy published baselines from the cited papers (same protocol only)."))

    abl = [("S", "–", "spatial"), ("F", "–", "frequency"), ("T", "–", "temporal"),
           ("S + F", "Stacking", "stacking_SF"), ("S + T", "Stacking", "stacking_ST"), ("F + T", "Stacking", "stacking_FT"),
           ("S + F + T", "Mean", "mean"), ("S + F + T", "Weighted mean", "wmean"),
           ("S + F + T", "Logistic regression", "logreg"), ("S + F + T", "Stacking, no OOF", "stacking_nooof"),
           ("S + F + T", "Stacking (proposed)", F)]
    sfx = F[len("stacking"):]                # "_valmeta" in the quick run: baselines carry the same suffix
    abl = [(c, f, m + sfx if m in ("stacking_SF", "stacking_ST", "stacking_FT", "mean", "wmean", "logreg") else m)
           for c, f, m in abl]
    rows = [[c, f, fmt(R.values(auc, "ffpp_c23", m, "test")), fmt(R.values(auc, "ffpp_c40", m, "test")),
             fmt(R.values(auc, "ffpp_c23", m, "celebdf")), fmt(R.robust_drop("ffpp_c23", m))] for c, f, m in abl]
    note = "DeLong p, proposed stacking vs each row (c23): " + ", ".join(
        f"{m} {R.delong('ffpp_c23', F, m)}" for _, _, m in abl[:-1])
    for name in ("oof", "insample"):         # leakage: validation vs test AUC of the meta-learner
        tag = "stacking" if name == "oof" else "stacking_nooof"
        note += f"\n\n{tag}: val AUC {fmt(R.values(auc, 'ffpp_c23', tag, 'val'))}, test AUC {fmt(R.values(auc, 'ffpp_c23', tag, 'test'))}"
    t.append(md("Table 7.4: Cue subsets and fusion rules, AUC (%)",
                ["Cues", "Fusion", "FF++ c23", "FF++ c40", "Celeb-DF v2", "Robust drop"], rows, note))

    det = [("Haar cascade (earlier prototype)", "ffpp_c23_haar"), ("MTCNN", "ffpp_c23_mtcnn"),
           ("RetinaFace (proposed)", "ffpp_c23")]
    rows = [[n, "Spatial", fmt(R.values(auc, s, "spatial", "test")), detection_failure(a.data, s), "–"] for n, s in det]
    tmp = [("T = 8 consecutive frames", "ffpp_c23", "temporal_T8"), ("T = 16 (proposed)", "ffpp_c23", "temporal"),
           ("T = 32 consecutive frames", "ffpp_c23_clip32", "temporal_T32"),
           ("T = 16, no frame-difference stream", "ffpp_c23", "temporal_nodiff")]
    rows += [[n, "Temporal", fmt(R.values(auc, s, m, "test")), "–", "–"] for n, s, m in tmp]
    t.append(md("Table 7.5: Design choices, AUC (%) on FF++ c23",
                ["Variant", "Affects", "AUC", "Detection failures (%)", "CPU s / video"], rows,
                "CPU time: run `python detector.py --input <10 s video>` and read the timing line."))

    rows = [["None", "–"] + [fmt(R.values(auc, "ffpp_c23", m, "test")) for m in models]]
    for kind, levels in LEVELS.items():
        rows += [[kind if i == 0 else "", lv] + [fmt(R.values(auc, "ffpp_c23", m, f"ffpp_c23_{kind}{lv}")) for m in models]
                 for i, lv in enumerate(levels)]
    t.append(md("Table 7.6: AUC (%) under post-processing on the FF++ c23 test split",
                ["Perturbation", "Level", "Spatial", "Frequency", "Temporal", "Fused"], rows))

    preds = [R.load("ffpp_c23", c, a.seeds[0], "test") for c in CUES]
    if all(p is not None for p in preds):
        res = pd.concat([(p.set_index("video_id").label - p.set_index("video_id").prob).abs().rename(c)
                         for p, c in zip(preds, CUES)], axis=1).dropna()
        corr = res.corr()
        rows = [[c] + [f"{corr.loc[c, d]:.2f}" if j >= i else "" for j, d in enumerate(CUES)] for i, c in enumerate(CUES)]
        t.append(md("Table 7.7: Pearson correlation of cue errors on FF++ c23 test", ["", *CUES], rows))

    if a.deletion:
        deletion(a.runs, a.data, a.deletion, a.out)
    dl = os.path.join(a.out, "deletion.json")
    dl = json.load(open(dl)) if os.path.exists(dl) else None
    fused = R.load("ffpp_c23", F, a.seeds[0], "test")
    if fused is not None:
        # decide as the deployed system does: threshold at the validation fake share (prior correction)
        calib = os.path.join(a.runs, "ffpp_c23", "fusion", F, f"seed{a.seeds[0]}", "calib.json")
        thr = json.load(open(calib)).get("prior", 0.5) if os.path.exists(calib) else 0.5
        err = lambda d: ((d.prob >= thr).astype(int) != d.label).astype(int)
        band_acc = lambda b: (lambda d: 1 - err(d[d.band == b]).mean() if (d.band == b).any() else np.nan)
        rows = [["Grad-CAM deletion", "Deletion AUC x100 (lower is better)",
                 f"{100 * dl['gradcam']:.1f}" if dl else "–", f"Random map: {100 * dl['random']:.1f}" if dl else "–"],
                ["Cue removal by Shapley rank", "Mean change in p_fused x100, top-ranked cue removed",
                 fmt(R.values(lambda d: d.d_top.mean(), "ffpp_c23", F, "test")),
                 "Lowest-ranked cue removed: " + fmt(R.values(lambda d: d.d_bottom.mean(), "ffpp_c23", F, "test"))],
                ["Conflict as error signal", "AUROC of k for predicting a misclassification",
                 fmt(R.values(lambda d: roc_auc_score(err(d), d.conflict), "ffpp_c23", F, "test")), "Chance: 50"],
                ["Accuracy by conflict band", "Accuracy (%) at low / medium / high k",
                 " / ".join(fmt(R.values(band_acc(b), "ffpp_c23", F, "test")) for b in ("low", "medium", "high")), "–"]]
        t.append(md("Table 7.8: Faithfulness of explanations on FF++ c23 test", ["Test", "Measure", "This project", "Control"],
                    rows, risk_coverage(fused, a.out, thr) + " Figure 7.1: results/risk_coverage.png"))

    sizes = [os.path.getsize(p) for p in glob.glob("models/*.keras")]
    if sizes:
        t.append(md("Table 7.9 (partial): model size", ["Item", "Value"],
                    [["Model size on disk (all branches and meta-learner)", f"{sum(sizes) / 2 ** 20:.1f} MB"]]))

    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("".join(t))
    print("".join(t))
    print(f"\n[OK] {a.out}/tables.md")


if __name__ == "__main__":
    main()
