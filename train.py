"""Train one cue branch with out-of-fold predictions (Section 5.1.5, Fig. 5.2).

  # 1. five fold models -> out-of-fold predictions for every training video
  for k in 0 1 2 3 4; do python train.py --cue spatial --set data/ffpp_c23 --fold $k; done
  # 2. final model on the whole training split -> train / val / test predictions
  python train.py --cue spatial --set data/ffpp_c23 --fold full
  # 3. cross-dataset and robustness sets with the final model
  python train.py --cue spatial --set data/ffpp_c23 --fold full --predict data/celebdf data/ffpp_c23_jpeg30

Cues: spatial, frequency, temporal (+ temporal_T8, temporal_T32, temporal_nodiff).
Leave-one-method-out (Table 7.2): add --exclude-method DF|F2F|FS|NT.
Everything is written to runs/<set>[_no<METHOD>]/<cue>/seed<seed>/<fold|full>/.
A finished model is reused, so an interrupted Colab session can simply be rerun.
"""
import argparse
import os

import data


def cue_module(cue):
    if cue.startswith("temporal"):
        import temporal as m
    else:
        import image_cues as m
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cue", required=True,
                    choices=["spatial", "frequency", "temporal", "temporal_T8", "temporal_T32", "temporal_nodiff"])
    ap.add_argument("--set", required=True, help="training data set, e.g. data/ffpp_c23")
    ap.add_argument("--fold", required=True, choices=["0", "1", "2", "3", "4", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-method", choices=["DF", "F2F", "FS", "NT"])
    ap.add_argument("--train-frames", type=int, default=None,
                    help="uniform frames per training video for image cues (default: all 32)")
    ap.add_argument("--predict", nargs="*", default=[], help="extra data sets to score with the full model")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()

    m = cue_module(a.cue)
    out = os.path.join(a.runs, data.run_name(a.set, a.exclude_method), a.cue, f"seed{a.seed}",
                       "full" if a.fold == "full" else f"fold{a.fold}")
    os.makedirs(out, exist_ok=True)

    train_v = data.load_videos(a.set, "train", a.exclude_method)
    val_v = data.load_videos(a.set, "val", a.exclude_method)
    if a.fold == "full":
        fit_v = train_v
        to_predict = {"train": train_v, "val": val_v, "test": data.load_videos(a.set, "test")}
    else:
        fit_v, held = data.fold_split(train_v, int(a.fold))
        to_predict = {"oof": held}
    print(f"[INFO] {a.cue}: training on {len(fit_v)} videos, validating on {len(val_v)} -> {out}")

    if not os.path.exists(os.path.join(out, "model.keras")):
        m.train(a.cue, fit_v, val_v, out, seed=a.seed, train_frames=a.train_frames)
    model = m.load(a.cue, out)

    to_predict.update({os.path.basename(os.path.normpath(s)): data.load_videos(s) for s in a.predict})
    for name, videos in to_predict.items():
        path = os.path.join(out, f"pred_{name}.csv")
        if not os.path.exists(path):
            data.write_preds(path, *m.predict(a.cue, model, videos))
            print(f"[OK] {path}")


if __name__ == "__main__":
    main()
