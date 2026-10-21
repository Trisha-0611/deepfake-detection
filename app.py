"""Streamlit interface (Chapter 6). Run with:  streamlit run app.py"""
import os
import tempfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deepfake detection", layout="centered")

CUES = [("spatial", "Spatial", "EfficientNet-B0 on face crops"),
        ("frequency", "Frequency", "MobileNetV2 on FFT spectra"),
        ("temporal", "Temporal", "BiLSTM on 16-frame clips")]


@st.cache_resource(show_spinner="Loading models…")
def load_detector():
    from detector import DeepfakeDetector
    return DeepfakeDetector(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))


def render_verdict(r):
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", r["verdict"])
    c2.metric("Confidence", f"{r['confidence']:.0%}")
    c3.metric("Cue conflict", f"{r['band']} (k = {r['conflict']:.2f})")
    if r["band"] == "high":
        st.warning("The cues disagree. Treat this verdict with caution.")
    st.caption("Confidence assumes a video is equally likely to be real or fake. Cue conflict is low, medium or "
               "high relative to the validation videos.")

    rows = [{"Cue": name, "Model": model,
             "P(fake)": f"{r['probs'][key]:.2f}" if key in r["probs"] else "not available",
             "Contribution φ": f"{r['shapley'][key]:+.2f}" if key in r["shapley"] else "–"}
            for key, name, model in CUES]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("φ is each cue's Shapley contribution to the fused score (log-odds): positive pushes towards FAKE, "
               "negative towards REAL.")


def render_evidence(r):
    c1, c2 = st.columns(2)
    c1.image(r["face"], caption="Aligned face (highest spatial score)")
    c2.image(r["gradcam"], caption=f"Grad-CAM, peak on {r['cam_region']}")
    st.markdown("**Frequency bands**: drop in P(fake) when each band is replaced by a real-face spectrum "
                "(1 = lowest frequency)")
    st.bar_chart(pd.DataFrame({"importance": r["band_scores"]}, index=range(1, 9)))
    if "frame_importance" in r:
        st.markdown("**Frames**: importance of each frame in the most suspicious 16-frame clip")
        st.line_chart(pd.DataFrame({"importance": r["frame_importance"]}, index=r["frame_indices"]))


def render_report(r):
    st.write(r["report"])
    st.download_button("Download report", r["report"], file_name="deepfake_report.txt", on_click="ignore")
    t = r["timing"]
    st.caption(f"Face found in {r['n_found']} of {r['n_sampled']} sampled frames · face pipeline "
               f"{t['preprocessing']:.1f} s · cues {t['cues']:.1f} s · fusion and explanation "
               f"{t['fusion_explanation']:.1f} s")


def analyse_file(uploaded):
    try:
        detector = load_detector()
    except FileNotFoundError as e:
        st.error(f"No trained models found. {e}")
        return
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())       # getvalue(): the preview may already have read the stream
        tmp_path = tmp.name
    try:
        with st.status("Analysing…", expanded=True) as status:     # stages appear as they start (Section 6.2)
            r = detector.detect(tmp_path, progress=st.write)
            status.update(label="Analysis failed" if "error" in r else "Analysis complete",
                          state="error" if "error" in r else "complete", expanded=False)
    finally:
        os.unlink(tmp_path)                  # NFR4: no uploaded media is kept
    if "error" in r:
        st.error(r["error"])
        return
    verdict_tab, evidence_tab, report_tab = st.tabs(["Verdict", "Evidence", "Report"])
    with verdict_tab:
        render_verdict(r)
    with evidence_tab:
        render_evidence(r)
    with report_tab:
        render_report(r)


def main():
    st.title("Deepfake detection")
    st.caption("Spatial, frequency and temporal cues with an explained fusion. The result is evidence for a "
               "human decision, not proof. Uploads are deleted after analysis.")
    uploaded = st.file_uploader("Video or face image", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    if uploaded is None:
        return
    is_image = os.path.splitext(uploaded.name)[1].lower() in {".jpg", ".jpeg", ".png"}
    (st.image if is_image else st.video)(uploaded)
    if st.button("Analyse", type="primary"):
        analyse_file(uploaded)


if __name__ == "__main__":
    main()
