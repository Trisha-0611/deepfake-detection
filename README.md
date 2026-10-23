# Explainable Multi-Cue Deepfake Detection

Code for the B.Tech. BCSE497J project report *Explainable Multi-Cue Deepfake Detection: Fusing
Spatial, Frequency and Temporal Evidence with Faithful Cue-Level Attribution*.

| File | Report section |
|---|---|
| `face_pipeline.py` | 5.1.1 RetinaFace detection, IoU tracking, landmark smoothing, ArcFace alignment (A.1) |
| `prepare_data.py` | 5.2.1 FF++ / Celeb-DF v2 / DFDC manifests with the official splits; crop extraction; robustness sets |
| `image_cues.py` | 5.1.2 spatial cue (EfficientNet-B0), 5.1.3 frequency cue (MobileNetV2 on log-FFT) |
| `temporal.py` | 5.1.4 temporal cue: 16 consecutive frames, frame-difference stream, BiLSTM + attention (A.2) |
| `train.py` | Fig. 5.2 out-of-fold training of any cue, leave-one-method-out |
| `stacking.py` | 5.1.5 meta-learner on out-of-fold logits, temperature scaling, fusion baselines (A.3) |
| `explain.py` | 5.1.6 exact Shapley values, conflict index, Grad-CAM, band occlusion, deletion test (A.4) |
| `perturb.py`, `delong.py` | 5.2.4 perturbations (A.5), DeLong test (A.6) |
| `evaluate.py` | Chapter 7: Tables 7.1–7.8 and Fig. 7.1 from the saved predictions |
| `detector.py`, `app.py` | Chapter 6: command-line and Streamlit analysis of one file |
| `test_explain.py` | Table 5.3, TC4–TC7 |

## Branches

| Branch | Contents |
|---|---|
| `main` | Current system (same as `multi-cue`) |
| `multi-cue` | Explainable multi-cue detector described in the report |
| `prototype` | Earlier prototype (Haar cascade, demo data), kept for reference |

## 1. Datasets

All three are free for research but need an access request. Request them first; approval takes days.

**FaceForensics++ (training and in-dataset tests).**
1. Fill in the access form linked from <https://github.com/ondyari/FaceForensics>. You receive a download script by e-mail.
2. Download the videos of the originals and the four methods, at c23 and c40 (not `-d all`, which adds
   FaceShifter and the DeepFakeDetection actor set):
   ```bash
   for d in original Deepfakes Face2Face FaceSwap NeuralTextures; do
     python download-FaceForensics.py /content/FF -d $d -c c23 -t videos
     python download-FaceForensics.py /content/FF -d $d -c c40 -t videos
   done
   ```
3. Put the official split files `train.json`, `val.json`, `test.json` from
   <https://github.com/ondyari/FaceForensics/tree/master/dataset/splits> in `/content/FF/splits/`.

**Celeb-DF v2 (cross-dataset test only).** Request access at
<https://github.com/yuezunli/celeb-deepfakeforensics>. The download contains `Celeb-real/`,
`Celeb-synthesis/`, `YouTube-real/` and `List_of_testing_videos.txt` (518 test videos); only the
test list is used.

**DFDC (cross-dataset test only).** Use the public test set from <https://ai.meta.com/datasets/dfdc/>
(do not download the 470 GB training set). `prepare_data.py dfdc` reads either a `labels.csv`
(filename, label) or Kaggle-style `metadata.json` files found under `--root`. If you instead use one of
the Kaggle competition parts (<https://www.kaggle.com/c/deepfake-detection-challenge/data>), say so in
Section 5.2.1 of the report.

### Without an access request (unofficial Kaggle copies)

Needs only a Kaggle account and an API token (kaggle.com → Settings → API → Create New Token → `kaggle.json`).
These are third-party re-uploads, not the authors' releases: say so in Section 5.2.1 and drop the
"signed agreements" wording from Sections 3.2.1 and 7.9.

```bash
pip install -q kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# FF++ c23 videos (~18 GB; extra FaceShifter / DeepFakeDetection folders are ignored)
kaggle datasets download -d xdxd003/ff-c23 -p /content/FF --unzip
# official split files are public
mkdir -p /content/FF/splits && for s in train val test; do
  curl -sSL -o /content/FF/splits/$s.json https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$s.json; done
# Celeb-DF v2 (~10 GB)
kaggle datasets download -d reubensuju/celeb-df-v2 -p /content/Celeb-DF-v2 --unzip
# DFDC: open kaggle.com/c/deepfake-detection-challenge/rules and click "I understand and accept" (instant),
# then take the labelled 400-video training sample (train_sample_videos/ with metadata.json)
kaggle competitions download -c deepfake-detection-challenge -p /content/dfdc && unzip -q /content/dfdc/*.zip -d /content/dfdc
```

`prepare_data.py` finds FF++ videos in either layout. Use `--root` on the folder that contains the
method folders, and pass `--splits-dir /content/FF/splits`. If `List_of_testing_videos.txt` ends up in a
subfolder of the Celeb-DF download, point `--root` at that subfolder. In `/content/dfdc`, the unlabelled
`test_videos/` are skipped automatically.

The Kaggle FF++ copy has c23 only. For the c40 columns, either drop them or re-encode c23 and label the
result "c23 re-encoded at CRF 40 (approximation of c40)":
```bash
cd /content/FF && find FaceForensics++_C23 -name '*.mp4' | while read f; do
  mkdir -p "c40approx/$(dirname "$f")"; ffmpeg -loglevel error -n -i "$f" -c:v libx264 -crf 40 -an "c40approx/$f"; done
python prepare_data.py ffpp --root /content/FF/c40approx --splits-dir /content/FF/splits --out data/ffpp_c40
```

Do not use randomuser.me or thispersondoesnotexist images: they are not video deepfakes, and the old
demo set put the same photos in both classes.

## 2. Training on Colab

Work on the Colab local disk (`/content`), not on Drive, and copy `data/` and `runs/` to Drive at the end
of each session. Install the same versions as `requirements.txt` so the models load in the app:

```bash
pip install -q "tensorflow[and-cuda]==2.20.0" keras==3.13.2 tf-keras==2.20.1 retina-face==0.0.19 \
    opencv-python-headless==4.10.0.84 scikit-learn scipy pandas matplotlib
```

Every step skips work that is already done, so an interrupted session can be rerun with the same command.

```bash
# 1. aligned face crops (32 uniform frames + 8 clips of 16 frames per video)
python prepare_data.py ffpp --root /content/FF --compression c23 --out data/ffpp_c23
python prepare_data.py ffpp --root /content/FF --compression c40 --out data/ffpp_c40
python prepare_data.py celebdf --root /content/Celeb-DF-v2 --out data/celebdf
python prepare_data.py dfdc --root /content/dfdc_test --out data/dfdc
#    robustness sets, named ffpp_c23_<kind><level> (levels in perturb.py)
PERT="jpeg:90 jpeg:70 jpeg:50 jpeg:30 resize:0.75 resize:0.5 resize:0.25 noise:5 noise:10 noise:20 blur:1 blur:2 blur:3"
for p in $PERT; do
  python prepare_data.py ffpp --root /content/FF --compression c23 --split test --perturb $p --out data/ffpp_c23_${p/:/}
done
SETS="celebdf dfdc"; for p in $PERT; do SETS="$SETS ffpp_c23_${p/:/}"; done
DIRS=""; for s in $SETS; do DIRS="$DIRS data/$s"; done

# 2. temporal features (frozen EfficientNet-B0, once per data set)
python temporal.py --features data/ffpp_c23 data/ffpp_c40 $DIRS

# 3. each cue: 5 fold models (out-of-fold predictions) + final model with predictions on every test set
for cue in spatial frequency temporal; do
  for k in 0 1 2 3 4; do python train.py --cue $cue --set data/ffpp_c23 --fold $k; done
  python train.py --cue $cue --set data/ffpp_c23 --fold full --predict $DIRS
done

# 4. fusion, then export the deployable models
python stacking.py --run runs/ffpp_c23 --eval val test $SETS --export models

# 5. tables
python evaluate.py --seeds 0 --deletion 200
```

Repeat steps 3–4 with `--seed 1` and `--seed 2` for mean ± std, and with `--set data/ffpp_c40` (and
`--run runs/ffpp_c40`) for the c40 columns.

**Ablations and generalisation (Tables 7.2, 7.4, 7.5)**

```bash
# cue subsets and fusion rules (reuse the step-3 predictions)
for c in spatial,frequency spatial,temporal frequency,temporal; do python stacking.py --run runs/ffpp_c23 --cues $c --eval val test $SETS; done
for f in mean wmean logreg; do python stacking.py --run runs/ffpp_c23 --fusion $f --eval val test $SETS; done
python stacking.py --run runs/ffpp_c23 --meta-source insample --eval val test $SETS     # "Stacking, no OOF"

# leave one method out
for m in DF F2F FS NT; do
  for cue in spatial frequency temporal; do
    for k in 0 1 2 3 4; do python train.py --cue $cue --set data/ffpp_c23 --exclude-method $m --fold $k; done
    python train.py --cue $cue --set data/ffpp_c23 --exclude-method $m --fold full
  done
  python stacking.py --run runs/ffpp_c23_no$m
done

# temporal design choices
for cue in temporal_T8 temporal_nodiff; do python train.py --cue $cue --set data/ffpp_c23 --fold full; done
python prepare_data.py ffpp --root /content/FF --compression c23 --clip-len 32 --out data/ffpp_c23_clip32
python temporal.py --features data/ffpp_c23_clip32
python train.py --cue temporal_T32 --set data/ffpp_c23_clip32 --fold full

# face detector (spatial cue only)
for d in haar mtcnn; do
  python prepare_data.py ffpp --root /content/FF --compression c23 --detector $d --out data/ffpp_c23_$d
  python train.py --cue spatial --set data/ffpp_c23_$d --fold full
done
```

**Compute budget.** One image-cue model on FF++ c23 takes roughly 1–2 h on a T4. The full plan is
several hundred runs, more than a free Colab account allows in a semester. Cheaper options:
`--train-frames 8` for the image cues, the leave-one-method-out runs with `stacking.py --meta-source val`
instead of five extra folds each, and one seed for the ablations. Whatever you cut, change Sections 5.2.2
and 3.2.1 of the report to match.

## 3. Running the analyst interface

```bash
pip install -r requirements.txt
streamlit run app.py
python detector.py --input video.mp4      # prints the report and the CPU timing for Table 7.9
python test_explain.py                    # TC4–TC7
```

`models/` must contain the files written by `stacking.py --export models`.
