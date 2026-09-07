"""
app.py
------
Stage 6: Streamlit web demo for the deepfake detection system.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =========================================================================== #
#  Page config  –  must be FIRST
# =========================================================================== #
st.set_page_config(
    page_title="DEEPFAKE DETECTION",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================================== #
#  CSS — mature, professional dark theme
# =========================================================================== #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── reset ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #080c14 !important;
    color: #c9d1d9 !important;
}

/* ── layout ──────────────────────────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1300px !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── topbar ──────────────────────────────────────────────────── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 28px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 32px;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-logo {
    width: 36px; height: 36px;
    background: #1a2236;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.topbar-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6366f1;
    letter-spacing: -0.3px;
}
.topbar-tag {
    font-size: 0.72rem;
    color: #484f58;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin-top: 1px;
}
.topbar-badges {
    display: flex;
    gap: 8px;
}
.badge {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #7d8590;
    letter-spacing: 0.3px;
}

/* ── upload panel ────────────────────────────────────────────── */
.panel {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 14px;
}
.panel-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #484f58;
    text-transform: uppercase;
    margin-bottom: 16px;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
    padding: 6px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.4) !important;
}

/* ── sliders ─────────────────────────────────────────────────── */
[data-testid="stSlider"] { padding: 0; }
[data-testid="stSlider"] .stSlider > div > div > div > div {
    background: #4f46e5 !important;
}

/* ── analyse button ──────────────────────────────────────────── */
.stButton > button {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    width: 100% !important;
    letter-spacing: 0.8px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #4338ca !important;
}

/* ── verdict strip ───────────────────────────────────────────── */
.verdict-strip {
    border-radius: 10px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
}
.verdict-strip-fake {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
}
.verdict-strip-real {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.22);
}
.verdict-strip-uncertain {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.22);
}
.verdict-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.verdict-main {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1;
}
.verdict-meta {
    font-size: 0.82rem;
    color: #7d8590;
    margin-top: 6px;
}
.verdict-score-block {
    text-align: right;
}
.verdict-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}
.verdict-score-label {
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #484f58;
    margin-top: 4px;
}

/* ── cue rows ────────────────────────────────────────────────── */
.cue-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.cue-row:last-child { border-bottom: none; }
.cue-name-col { min-width: 200px; }
.cue-name {
    font-size: 0.84rem;
    font-weight: 500;
    color: #e6edf3;
}
.cue-model {
    font-size: 0.7rem;
    color: #484f58;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.cue-bar-col { flex: 1; }
.cue-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.cue-bar-fill { height: 100%; border-radius: 4px; }
.cue-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    min-width: 52px;
    text-align: right;
}
.cue-sev {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 5px;
    min-width: 72px;
    text-align: center;
}

/* ── explanation ─────────────────────────────────────────────── */
.summary-block {
    border-left: 3px solid;
    padding: 12px 18px;
    border-radius: 0 8px 8px 0;
    background: rgba(255,255,255,0.025);
    margin-bottom: 20px;
    font-size: 0.875rem;
    line-height: 1.7;
    color: #8b949e;
}
.evidence-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.ev-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.ev-title { font-size: 0.88rem; font-weight: 600; color: #e6edf3; }
.ev-detail { font-size: 0.82rem; color: #7d8590; line-height: 1.65; }
.ev-tip {
    margin-top: 10px;
    padding: 8px 12px;
    background: rgba(79,70,229,0.07);
    border-radius: 6px;
    font-size: 0.76rem;
    color: #6366f1;
}

/* ── section heading ─────────────────────────────────────────── */
.section-head {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #484f58;
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── stat cards ──────────────────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 4px;
}
.stat-card {
    flex: 1;
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 18px;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #484f58;
    margin-top: 4px;
    letter-spacing: 0.5px;
}

/* ── placeholder ─────────────────────────────────────────────── */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 380px;
    border: 1px dashed rgba(255,255,255,0.07);
    border-radius: 12px;
    color: #21262d;
    gap: 14px;
}
.placeholder-icon {
    font-size: 2.2rem;
    opacity: 0.4;
}
.placeholder-text {
    font-size: 0.88rem;
    color: #30363d;
    letter-spacing: 0.3px;
}

/* ── dataframe + expander ────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.01) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── metrics ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #484f58 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================================== #
#  Detector  –  cached
# =========================================================================== #

@st.cache_resource(show_spinner="Loading models…")
def load_detector(threshold: float):
    try:
        from detector import DeepfakeDetector
        return DeepfakeDetector(threshold=threshold)
    except SystemExit:
        return None
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None


# =========================================================================== #
#  Explainability engine  (unchanged logic)
# =========================================================================== #

def generate_explanation(result: dict) -> dict:
    p_s    = result.get("spatial_prob",   0.5)
    p_f    = result.get("frequency_prob", 0.5)
    p_t    = result.get("temporal_prob",  0.5)
    p_fuse = result.get("fusion_prob",    0.5)
    verdict = result.get("verdict", "UNCERTAIN")

    def severity(p):
        if p > 0.75: return "high"
        if p > 0.55: return "medium"
        return "low"

    def sev_colour(s):
        return {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}.get(s, "#6b7280")

    def sev_label(s):
        return {"high": "Strong", "medium": "Moderate", "low": "Weak"}.get(s, "")

    cues = []

    sev = severity(p_s)
    if verdict == "FAKE":
        spatial_texts = {
            "high":   ("Face-swap blending artifacts",
                       "EfficientNetB0 detected strong visual inconsistencies in the face region. "
                       "Common signs include unnatural skin tone gradients, sharp boundary edges "
                       "around the face, and texture mismatches between the face and background "
                       "— all characteristic of GAN-based face-swap methods."),
            "medium": ("Possible visual manipulation",
                       "Some visual cues suggest manipulation. Texture patterns in the face "
                       "region show mild statistical differences from authentic faces."),
            "low":    ("Minimal spatial evidence",
                       "The spatial model found little visual evidence of manipulation. "
                       "Other cues carry more weight in this verdict."),
        }
    else:
        spatial_texts = {
            "high":   ("Natural face texture",
                       "No visual artifacts detected. Skin texture, lighting, and facial boundaries "
                       "all appear consistent with a genuine face."),
            "medium": ("Likely natural", "Minor inconsistencies may be due to compression or image quality."),
            "low":    ("Uncertain", "Spatial features do not strongly indicate manipulation."),
        }
    title, detail = spatial_texts[sev]
    cues.append({"name": "Spatial", "model": "EfficientNetB0", "prob": p_s,
                 "sev": sev, "colour": sev_colour(sev), "slabel": sev_label(sev),
                 "title": title, "detail": detail,
                 "what": "Blurry face edges, mismatched skin tone, unnatural eye reflections."})

    sev = severity(p_f)
    if verdict == "FAKE":
        freq_texts = {
            "high":   ("GAN fingerprints in frequency domain",
                       "The FFT log-magnitude spectrum shows significant anomalies. GAN-generated "
                       "images leave characteristic checkerboard patterns caused by convolutional "
                       "upsampling — invisible to the eye but detectable spectrally."),
            "medium": ("Frequency domain irregularities",
                       "Moderate spectral anomalies detected. Some frequency components deviate "
                       "from natural distributions."),
            "low":    ("Minor spectral differences",
                       "Only subtle frequency differences found. This cue contributes little to the verdict."),
        }
    else:
        freq_texts = {
            "high":   ("Clean frequency spectrum",
                       "No GAN fingerprints or checkerboard artifacts found. The frequency domain "
                       "matches distributions typical of natural camera images."),
            "medium": ("Mostly natural spectrum", "Minor spectral variation likely due to JPEG compression."),
            "low":    ("Uncertain", "Frequency evidence is inconclusive."),
        }
    title, detail = freq_texts[sev]
    cues.append({"name": "Frequency", "model": "FFT + MobileNetV2", "prob": p_f,
                 "sev": sev, "colour": sev_colour(sev), "slabel": sev_label(sev),
                 "title": title, "detail": detail,
                 "what": "Grid/checkerboard patterns in FFT spectrum (invisible in RGB)."})

    sev = severity(p_t)
    if verdict == "FAKE":
        temp_texts = {
            "high":   ("Temporal flickering detected",
                       "The BiLSTM detected unnatural patterns across the frame sequence. "
                       "Deepfake methods produce frame-to-frame jitter — flickering facial features, "
                       "inconsistent blinking rates, and abrupt texture changes between frames."),
            "medium": ("Mild temporal irregularities",
                       "Some inter-frame inconsistency detected. May indicate partial manipulation "
                       "or lossy video compression artifacts."),
            "low":    ("Weak temporal signal",
                       "Temporal analysis is less decisive. The sequence may be too short or "
                       "manipulation too subtle for clear detection."),
        }
    else:
        temp_texts = {
            "high":   ("Natural facial dynamics",
                       "Face motion is smooth and consistent across frames. Blinking, "
                       "micro-expressions, and head movement follow natural temporal patterns."),
            "medium": ("Mostly consistent motion", "Minor frame variation within normal range."),
            "low":    ("Uncertain", "Temporal evidence is inconclusive."),
        }
    title, detail = temp_texts[sev]
    cues.append({"name": "Temporal", "model": "BiLSTM", "prob": p_t,
                 "sev": sev, "colour": sev_colour(sev), "slabel": sev_label(sev),
                 "title": title, "detail": detail,
                 "what": "Flickering edges, inconsistent blinking, sudden texture changes."})

    order = {"high": 0, "medium": 1, "low": 2}
    cues.sort(key=lambda c: order[c["sev"]])
    primary = max(cues, key=lambda c: c["prob"])

    if verdict == "FAKE":
        if p_fuse > 0.80:
            summary = (f"All three detection pathways converge: this media has been synthetically "
                       f"manipulated. The meta-learner's fusion score ({p_fuse:.1%}) reflects "
                       f"strong agreement across all cues.")
        elif p_fuse > 0.60:
            summary = (f"The fusion model returned a FAKE verdict ({p_fuse:.1%}). "
                       f"The strongest signal came from the {primary['name']} detector.")
        else:
            summary = (f"A marginal FAKE verdict ({p_fuse:.1%}). Evidence is present but not "
                       f"overwhelming — consider reviewing the source carefully.")
    elif verdict == "REAL":
        summary = (f"All three cues are consistent with authentic media (fusion score {p_fuse:.1%}). "
                   f"No manipulation signatures were detected.")
    else:
        summary = "The detectors disagree. This result is inconclusive."

    return {"verdict": verdict, "cues": cues, "primary": primary,
            "summary": summary, "p_fuse": p_fuse}


# =========================================================================== #
#  Render helpers
# =========================================================================== #

def _accent(verdict):
    return {"FAKE": "#ef4444", "REAL": "#22c55e"}.get(verdict, "#f59e0b")


def render_verdict(result: dict) -> None:
    verdict  = result.get("verdict", "UNCERTAIN")
    p_fuse   = result.get("fusion_prob", 0.5)
    conf     = result.get("confidence", 0.0)
    acc      = _accent(verdict)

    strip_cls = {"FAKE": "verdict-strip-fake",
                 "REAL": "verdict-strip-real"}.get(verdict, "verdict-strip-uncertain")
    label_txt = {"FAKE": "Manipulated Media Detected",
                 "REAL": "Authentic Media",
                 "UNCERTAIN": "Inconclusive"}.get(verdict, verdict)
    sub_txt   = {"FAKE": f"Confidence {conf:.1%}  ·  Fusion score {p_fuse:.3f}",
                 "REAL": f"Confidence {conf:.1%}  ·  Fusion score {p_fuse:.3f}",
                 "UNCERTAIN": "Detectors disagree — manual review recommended"}.get(verdict, "")

    st.markdown(f"""
    <div class="verdict-strip {strip_cls}">
        <div>
            <div class="verdict-label" style="color:{acc};">{verdict}</div>
            <div class="verdict-main" style="color:{acc};">{label_txt}</div>
            <div class="verdict-meta">{sub_txt}</div>
        </div>
        <div class="verdict-score-block">
            <div class="verdict-score-num" style="color:{acc};">{p_fuse:.1%}</div>
            <div class="verdict-score-label">Fusion Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_cue_table(result: dict) -> None:
    cues = [
        ("Spatial",    "EfficientNetB0",       result.get("spatial_prob",   0.5), "#6366f1"),
        ("Frequency",  "FFT + MobileNetV2",    result.get("frequency_prob", 0.5), "#8b5cf6"),
        ("Temporal",   "BiLSTM",               result.get("temporal_prob",  0.5), "#a78bfa"),
        ("Fusion",     "Meta-Learner MLP",     result.get("fusion_prob",    0.5), "#818cf8"),
    ]
    verdict = result.get("verdict", "UNCERTAIN")

    rows_html = ""
    for name, model, prob, colour in cues:
        pct     = int(prob * 100)
        is_fake = verdict == "FAKE"
        sev     = "Strong" if prob > 0.75 else "Moderate" if prob > 0.55 else "Weak"
        if is_fake:
            sev_bg   = {"Strong": "rgba(239,68,68,0.12)",  "Moderate": "rgba(245,158,11,0.12)", "Weak": "rgba(34,197,94,0.12)"}[sev]
            sev_col  = {"Strong": "#ef4444", "Moderate": "#f59e0b", "Weak": "#22c55e"}[sev]
        else:
            sev_bg   = "rgba(34,197,94,0.12)"
            sev_col  = "#22c55e"
            sev      = "Clean" if prob < 0.45 else "Low"

        rows_html += f"""
        <div class="cue-row">
            <div class="cue-name-col">
                <div class="cue-name">{name}</div>
                <div class="cue-model">{model}</div>
            </div>
            <div class="cue-bar-col">
                <div class="cue-bar-bg">
                    <div class="cue-bar-fill" style="width:{pct}%; background:{colour};"></div>
                </div>
            </div>
            <div class="cue-pct" style="color:{colour};">{prob:.1%}</div>
            <div class="cue-sev" style="background:{sev_bg}; color:{sev_col};">{sev}</div>
        </div>"""

    st.markdown(f'<div class="panel"><p class="section-head">Cue Breakdown</p>{rows_html}</div>',
                unsafe_allow_html=True)


def render_explanation(result: dict) -> None:
    exp     = generate_explanation(result)
    verdict = exp["verdict"]
    acc     = _accent(verdict)

    st.markdown(f"""
    <p class="section-head" style="margin-top:24px;">Analysis Rationale</p>
    <div class="summary-block" style="border-color:{acc};">
        {exp['summary']}
    </div>
    """, unsafe_allow_html=True)

    for cue in exp["cues"]:
        sev_bg  = {"Strong": "rgba(239,68,68,0.1)",  "Moderate": "rgba(245,158,11,0.1)",
                   "Weak":   "rgba(34,197,94,0.1)",  "Clean":    "rgba(34,197,94,0.1)",
                   "Low":    "rgba(34,197,94,0.1)"}.get(cue["slabel"], "rgba(255,255,255,0.05)")

        st.markdown(f"""
        <div class="evidence-card">
            <div class="ev-header">
                <div>
                    <div class="ev-title">{cue['name']} — {cue['title']}</div>
                    <div style="font-size:0.72rem; color:#484f58; margin-top:2px;
                                font-family:'JetBrains Mono',monospace;">{cue['model']}</div>
                </div>
                <div style="text-align:right;">
                    <span style="background:{sev_bg}; color:{cue['colour']};
                                 border-radius:5px; padding:3px 10px;
                                 font-size:0.68rem; font-weight:600; letter-spacing:1px;">
                        {cue['slabel'].upper()}
                    </span>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem;
                                font-weight:600; color:{cue['colour']}; margin-top:4px;">
                        {cue['prob']:.1%}
                    </div>
                </div>
            </div>
            <div class="ev-detail">{cue['detail']}</div>
            <div class="ev-tip">What to look for: {cue['what']}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("How the fusion model combined these cues", expanded=False):
        p_s    = result.get("spatial_prob", 0.5)
        p_f    = result.get("frequency_prob", 0.5)
        p_t    = result.get("temporal_prob", 0.5)
        p_fuse = result.get("fusion_prob", 0.5)
        st.markdown(f"""
        <p style="font-size:0.85rem; color:#8b949e; line-height:1.7;">
        The <strong style="color:#e6edf3;">Meta-Learner MLP</strong> receives the three
        cue probabilities as a feature vector
        <code style="font-family:'JetBrains Mono',monospace; background:rgba(255,255,255,0.06);
              padding:2px 6px; border-radius:4px;">[{p_s:.3f}, {p_f:.3f}, {p_t:.3f}]</code>
        and passes them through two dense layers (32 → 16 units) to produce the final
        fusion score of <strong style="color:#818cf8;">{p_fuse:.1%}</strong>.<br><br>
        Unlike a simple average, the meta-learner learns to <em>down-weight unreliable
        cues</em> and <em>amplify cues that agree</em> — the stacking ensemble technique.
        </p>
        """, unsafe_allow_html=True)


def render_stats(result: dict) -> None:
    conf = result.get("confidence", 0.0)

    if "frames_analysed" in result:
        vals = [
            (str(result["frames_analysed"]), "Frames Analysed"),
            (str(result["faces_detected"]),  "Faces Detected"),
            (f"{conf:.1%}",                  "Confidence"),
            (f"{result.get('fusion_prob', 0.5):.3f}", "Fusion Score"),
        ]
    else:
        vals = [
            ("Yes" if result.get("face_detected") else "No", "Face Detected"),
            (f"{conf:.1%}",                                   "Confidence"),
            (f"{result.get('fusion_prob', 0.5):.3f}",         "Fusion Score"),
            (f"{result.get('spatial_prob', 0.5):.3f}",        "Spatial Score"),
        ]

    cards_html = "".join(
        f'<div class="stat-card"><div class="stat-val">{v}</div>'
        f'<div class="stat-lbl">{l}</div></div>'
        for v, l in vals
    )
    st.markdown(
        f'<p class="section-head" style="margin-top:24px;">Analysis Stats</p>'
        f'<div class="stat-row">{cards_html}</div>',
        unsafe_allow_html=True
    )


# =========================================================================== #
#  Main
# =========================================================================== #

def main() -> None:

    # ── Top bar ──────────────────────────────────────────────────────────── #
    st.markdown("""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="topbar-logo">🛡</div>
            <div>
                <div class="topbar-name">DEEPFAKE DETECTION</div>
                <div class="topbar-tag">Spatial · Frequency · Temporal · Fusion</div>
            </div>
        </div>
        <div class="topbar-badges">
            <span class="badge">EfficientNetB0</span>
            <span class="badge">MobileNetV2</span>
            <span class="badge">BiLSTM</span>
            <span class="badge">Meta-Learner</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout ───────────────────────────────────────────────────────────── #
    left, right = st.columns([1, 2], gap="large")

    with left:
        # Upload
        st.markdown('<div class="panel"><p class="panel-title">Input</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a video or image",
            type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Settings
        st.markdown('<div class="panel"><p class="panel-title">Settings</p>', unsafe_allow_html=True)
        fps = st.slider("Sample FPS", min_value=1, max_value=10, value=3,
                        help="Frames extracted per second (video only).")
        threshold = st.slider("Decision threshold", min_value=0.30, max_value=0.70,
                               value=0.50, step=0.01,
                               help="P(fake) above this → FAKE verdict.")
        st.markdown('</div>', unsafe_allow_html=True)

        analyse = st.button("ANALYSE", use_container_width=True)

        # Instructions
        st.markdown("""
        <div style="margin-top:14px; padding:16px 18px;
                    background:#0d1117; border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px; font-size:0.78rem; color:#484f58; line-height:1.9;">
            <div style="color:#7d8590; font-weight:600; margin-bottom:6px;
                        font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;">
                Instructions
            </div>
            1. Upload an <span style="color:#6366f1;">.mp4</span> video or face image<br>
            2. Adjust FPS and threshold if needed<br>
            3. Click <strong style="color:#e6edf3;">ANALYSE</strong><br>
            4. Review the cue breakdown and rationale
        </div>
        """, unsafe_allow_html=True)

    with right:

        if not analyse or uploaded is None:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-icon">🛡</div>
                <div class="placeholder-text">Upload a file and click Analyse to begin</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Running multi-cue analysis…"):
                    detector = load_detector(threshold)

                    if detector is None:
                        _models_dir = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "models")
                        _all = ["spatial_model.keras", "frequency_model.keras",
                                "temporal_model.keras", "fusion_model.keras"]
                        _found   = [f for f in _all if os.path.exists(os.path.join(_models_dir, f))]
                        _missing = [f for f in _all if not os.path.exists(os.path.join(_models_dir, f))]
                        st.warning("**No trained models found.** Train the models first.")
                        st.code(
                            "python train_spatial.py\n"
                            "python frequency_cue.py\n"
                            "python temporal_cue.py\n"
                            "python fusion.py\n"
                            "streamlit run app.py",
                            language="bash"
                        )
                        if _found:
                            st.success("Ready: " + ", ".join(f.replace("_model.keras","") for f in _found))
                        if _missing:
                            st.error("Missing: " + ", ".join(f.replace("_model.keras","") for f in _missing))
                        st.stop()

                    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                    if suffix.lower() in image_exts:
                        result = detector.detect_image(tmp_path)
                    else:
                        result = detector.detect_video(tmp_path, target_fps=fps)

            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if "error" in result and result.get("verdict") not in ("FAKE", "REAL"):
                st.error(result["error"])
                st.stop()

            # ── Results ───────────────────────────────────────────────── #
            render_verdict(result)
            render_cue_table(result)
            render_explanation(result)
            render_stats(result)

            # ── Per-frame table ───────────────────────────────────────── #
            per_frame = result.get("per_frame_results", [])
            if per_frame:
                st.markdown('<p class="section-head" style="margin-top:24px;">Per-Frame Detail</p>',
                            unsafe_allow_html=True)
                with st.expander(f"{len(per_frame)} frames analysed", expanded=False):
                    import pandas as pd
                    df = pd.DataFrame(per_frame)[["frame_idx", "spatial", "frequency"]].rename(
                        columns={"frame_idx": "Frame", "spatial": "Spatial P(fake)",
                                 "frequency": "Frequency P(fake)"})
                    st.dataframe(
                        df.style.background_gradient(
                            cmap="RdYlGn_r",
                            subset=["Spatial P(fake)", "Frequency P(fake)"],
                            vmin=0, vmax=1),
                        use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────── #
    st.markdown("""
    <div style="margin-top:60px; padding-top:20px;
                border-top:1px solid rgba(255,255,255,0.05);
                display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.72rem; color:#21262d;">
            DEEPFAKE DETECTION
        </span>
        <span style="font-size:0.72rem; color:#21262d; font-family:'JetBrains Mono',monospace;">
            Python 3.10 · TensorFlow 2.16 · Streamlit
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
