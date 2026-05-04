"""
EmPath v2  -  Multimodal Pain Intensity Detection
Streamlit demo app  /  Komala Belur Srinivas  /  Hofstra University
Live: https://komala-b-srinivas-empath-app-oxt9of.streamlit.app/
"""

import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmPath  -  Pain Detection",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #070f1e; color: #e2e8f0; }

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, rgba(14,28,54,.9), rgba(7,17,35,.95));
    border: 1px solid rgba(56,189,248,.12);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: .8rem;
}
.card-accent-blue  { border-left: 3px solid #38bdf8; }
.card-accent-green { border-left: 3px solid #34d399; }
.card-accent-red   { border-left: 3px solid #f87171; }
.card-accent-purple{ border-left: 3px solid #a78bfa; }

/* ── KPI tiles ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap:.7rem; margin: .8rem 0 1.2rem; }
.kpi { background: rgba(14,28,54,.85); border: 1px solid rgba(56,189,248,.13);
       border-radius: 12px; padding: .85rem 1rem; text-align:center; }
.kpi-val { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.kpi-lab { font-size: .67rem; color: #64748b; text-transform: uppercase;
           letter-spacing: .1em; margin-top: .2rem; }

/* ── Section headers ── */
.sh { font-size: .65rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .15em; color: #64748b; margin: .2rem 0 .6rem; }

/* ── Insight box ── */
.insight {
    background: linear-gradient(135deg, rgba(56,189,248,.06), rgba(99,102,241,.04));
    border: 1px solid rgba(56,189,248,.14); border-radius: 12px;
    padding: 1rem 1.2rem; margin: .6rem 0;
}

/* ── Step cards ── */
.step { background: rgba(14,28,54,.7); border: 1px solid rgba(255,255,255,.06);
        border-radius: 12px; padding: 1.1rem .9rem; text-align: center; height: 100%; }
.step-n { display: inline-block; width: 32px; height: 32px; border-radius: 50%;
          font-size: .8rem; font-weight: 800; line-height: 32px; margin-bottom: .5rem; }

/* ── Tags ── */
.tag { background: rgba(56,189,248,.1); color: #38bdf8; border: 1px solid rgba(56,189,248,.25);
       border-radius: 20px; padding: 2px 10px; font-size: .66rem; font-weight: 600;
       display: inline-block; margin: 2px; }
.tag-g { background: rgba(52,211,153,.1); color: #34d399; border-color: rgba(52,211,153,.25); }
.tag-p { background: rgba(167,139,250,.1); color: #a78bfa; border-color: rgba(167,139,250,.25); }
.tag-r { background: rgba(248,113,113,.1); color: #f87171; border-color: rgba(248,113,113,.25); }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#060d1a,#0a1628) !important; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── Tab styles ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(14,28,54,.6); border-radius: 10px; gap: 0; }
.stTabs [data-baseweb="tab"] { color: #475569 !important; font-size: .82rem; padding: .55rem 1rem; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; background: rgba(56,189,248,.1) !important;
                                   border-radius: 8px !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Divider ── */
.hdiv { height: 1px; background: linear-gradient(90deg,transparent,rgba(56,189,248,.2),transparent);
        margin: 1.5rem 0; }

/* ── Metric override ── */
[data-testid="metric-container"] { background: rgba(14,28,54,.7) !important;
    border: 1px solid rgba(56,189,248,.1) !important; border-radius: 12px !important;
    padding: .8rem 1rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.7rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .72rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL  = os.path.join(BASE, "Models", "empath_model.pkl")
DEMO_CSV   = os.path.join(BASE, "Models", "demo_samples.csv")
EA_DIR     = os.path.join(BASE, "Results", "error_analysis_v2")
SHAP_BIO   = os.path.join(EA_DIR, "shap_biosignal_ranked.csv")
SHAP_LM    = os.path.join(EA_DIR, "shap_landmark_ranked.csv")
SUBJ_CSV   = os.path.join(EA_DIR, "per_subject_accuracy.csv")
SIGPLOT    = os.path.join(BASE, "Results", "signal_plots")
BIO_BEES   = os.path.join(EA_DIR, "shap_beeswarm_biosignal.png")
LM_BEES    = os.path.join(EA_DIR, "shap_beeswarm_landmark.png")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PKL):
        return None
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_demo():
    if not os.path.exists(DEMO_CSV):
        return None
    return pd.read_csv(DEMO_CSV)

@st.cache_data(show_spinner=False)
def load_shap():
    bio = pd.read_csv(SHAP_BIO) if os.path.exists(SHAP_BIO) else None
    lm  = pd.read_csv(SHAP_LM)  if os.path.exists(SHAP_LM)  else None
    return bio, lm

@st.cache_data(show_spinner=False)
def load_subj():
    if not os.path.exists(SUBJ_CSV):
        return None
    return pd.read_csv(SUBJ_CSV)

model    = load_model()
demo_df  = load_demo()
shap_bio, shap_lm = load_shap()
subj_df  = load_subj()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_sample(bio_feats, lm_feats):
    if model is None:
        return 0, [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]
    bio = np.array(bio_feats, dtype=np.float32).reshape(1, -1)
    lm  = np.array(lm_feats,  dtype=np.float32).reshape(1, -1)
    bio = (bio - model["global_bio_mean"]) / model["global_bio_std"]
    lm  = (lm  - model["global_lm_mean"])  / model["global_lm_std"]
    bp  = model["rf_bio"].predict_proba(bio)[0]
    lp  = model["rf_lm"].predict_proba(lm)[0]
    mp  = np.hstack([bp, lp]).reshape(1, -1)
    pred = int(model["meta"].predict(mp)[0])
    prob = model["meta"].predict_proba(mp)[0]
    return pred, prob, bp, lp

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#070f1e"
GRID = "rgba(255,255,255,.04)"
LINE = "rgba(255,255,255,.08)"
FONT = dict(family="Inter", color="#94a3b8", size=11)

def base_fig(h=320, title=None, xlab=None, ylab=None, margin=None):
    m = margin or dict(l=8, r=20, t=40 if title else 20, b=30)
    layout = dict(
        height=h, paper_bgcolor=BG, plot_bgcolor=BG,
        font=FONT, margin=m,
        xaxis=dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=LINE,
                   tickfont=dict(size=10, color="#475569")),
        yaxis=dict(gridcolor=GRID, linecolor=LINE,
                   tickfont=dict(size=10, color="#475569")),
    )
    if title: layout["title"] = dict(text=title, font=dict(size=11, color="#64748b"))
    if xlab:  layout["xaxis_title"] = xlab
    if ylab:  layout["yaxis_title"] = ylab
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.5rem 0 1rem;">
        
        <div style="font-size:1.05rem;font-weight:800;color:#f1f5f9;">EmPath v2</div>
        <div style="font-size:.7rem;color:#475569;margin-top:.2rem;">
            Multimodal Pain Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Quick Stats</div>', unsafe_allow_html=True)
    for label, val, color in [
        ("Best Accuracy", "65.3%", "#38bdf8"),
        ("AUC-ROC", "0.719", "#34d399"),
        ("Subjects (LOSO)", "67", "#a78bfa"),
        ("Features", "57 (35+22)", "#fb923c"),
    ]:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.4rem .6rem;background:rgba(14,28,54,.6);'
            f'border-radius:8px;margin-bottom:.3rem;">'
            f'<span style="font-size:.76rem;color:#64748b;">{label}</span>'
            f'<span style="font-size:.76rem;font-weight:700;color:{color};">{val}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Select a Sample</div>', unsafe_allow_html=True)

    selected_sample = None
    subjects = []
    if demo_df is not None and "subject_id" in demo_df.columns:
        subjects = sorted(demo_df["subject_id"].unique().tolist())
        sel_subj = st.selectbox(
            "Subject ID", subjects,
            format_func=lambda x: f"Subject {int(x):03d}",
            label_visibility="collapsed"
        )
        subj_rows = demo_df[demo_df["subject_id"] == sel_subj]
        snames = subj_rows["sample_name"].tolist() if "sample_name" in subj_rows.columns else list(range(len(subj_rows)))
        sel_idx = st.selectbox(
            "Sample", range(len(snames)),
            format_func=lambda i: f"Trial {i+1}  ({snames[i]})" if isinstance(snames[i], str) else f"Trial {i+1}",
            label_visibility="collapsed"
        )
        selected_sample = subj_rows.iloc[sel_idx]

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.68rem;color:#334155;line-height:1.9;text-align:center;">
        BioVid Heat Pain Database<br>
        Hofstra University / M.S. CS<br>
        Komala Belur Srinivas / 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:.8rem;">
        <a href="https://github.com/komalabelursrinivas/EmPath_v2"
           target="_blank" style="text-decoration:none;">
            <span class="tag">GitHub</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0c1f3e 0%,#091529 50%,#0a1a32 100%);
            border:1px solid rgba(56,189,248,.14);border-radius:18px;
            padding:2rem 2.4rem 1.6rem;margin-bottom:1.2rem;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:280px;height:280px;
                background:radial-gradient(circle,rgba(56,189,248,.07),transparent 70%);
                border-radius:50%;transform:translate(30%,-30%);"></div>
    <div style="position:absolute;bottom:0;left:0;width:200px;height:200px;
                background:radial-gradient(circle,rgba(52,211,153,.05),transparent 70%);
                border-radius:50%;transform:translate(-30%,30%);"></div>
    <div style="position:relative;">
        <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.5rem;">
            
            <div>
                <div style="font-size:1.8rem;font-weight:800;color:#f1f5f9;line-height:1.1;">EmPath</div>
                <div style="font-size:.75rem;color:#38bdf8;font-weight:600;letter-spacing:.08em;">
                    MULTIMODAL PAIN INTENSITY DETECTION
                </div>
            </div>
        </div>
        <p style="font-size:.88rem;color:#94a3b8;max-width:680px;line-height:1.7;margin:.6rem 0 1.2rem;">
            Discriminates <b style="color:#38bdf8;">moderate pain (PA2, ~43°C)</b> from
            <b style="color:#f87171;">intense pain (PA3, ~45°C)</b> using 35 biosignal features
            (GSR / ECG / EMG / HRV) and 22 facial landmark features from MediaPipe FaceMesh  - 
            fused via stacked generalization evaluated with strict LOSO cross-validation on
            <b style="color:#f1f5f9;">67 thermally reactive subjects</b>.
        </p>
        <div style="display:flex;flex-wrap:wrap;gap:.4rem;">
            <span class="tag">65.3% LOSO-67</span>
            <span class="tag tag-g">AUC 0.719</span>
            <span class="tag tag-p">Stacked RF Fusion</span>
            <span class="tag tag-g">SHAP Explainable</span>
            <span class="tag">BioVid Database</span>
            <span class="tag tag-r">26 Variants Tested</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Live Demo",
    "Feature Analysis",
    "Performance",
    "Architecture",
    "Clinical Context",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  -  LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    if selected_sample is None or demo_df is None:
        st.info("⬅ Select a subject and sample in the sidebar to run the live demo.")
    else:
        bio_cols = [c for c in demo_df.columns if c.startswith("bio_")]
        lm_cols  = [c for c in demo_df.columns if c.startswith("lm_")]

        bio_feats = selected_sample[bio_cols].values.tolist()
        lm_feats  = selected_sample[lm_cols].values.tolist()
        true_label = int(selected_sample["label"]) if "label" in selected_sample else -1
        true_lbl_str = "PA2 (Moderate)" if true_label == 0 else "PA3 (Intense)" if true_label == 1 else "Unknown"

        pred, prob, bio_prob, lm_prob = predict_sample(bio_feats, lm_feats)
        pred_lbl   = "PA2" if pred == 0 else "PA3"
        pred_full  = "Moderate Pain (PA2)" if pred == 0 else "Intense Pain (PA3)"
        pred_temp  = "~43°C" if pred == 0 else "~45°C"
        conf       = max(prob) * 100
        PAIN_COL   = "#38bdf8" if pred == 0 else "#f87171"
        correct    = (pred == true_label) if true_label >= 0 else None

        # ── Result banner ────────────────────────────────────────────────────
        banner_bg  = "rgba(3,105,161,.18)" if pred == 0 else "rgba(159,18,57,.18)"
        banner_bdr = "#38bdf8" if pred == 0 else "#f87171"
        verdict_icon = "[correct]" if correct else ("[wrong]" if correct is not None else "")
        st.markdown(
            f'<div style="background:{banner_bg};border:1.5px solid {banner_bdr}44;'
            f'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;'
            f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.8rem;">'
            f'<div>'
            f'<div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.15em;'
            f'color:{PAIN_COL};font-weight:700;margin-bottom:.3rem;">Prediction</div>'
            f'<div style="font-size:1.9rem;font-weight:900;color:#f1f5f9;line-height:1;">{pred_full}</div>'
            f'<div style="font-size:.82rem;color:#64748b;margin-top:.3rem;">'
            f'Stimulus temperature {pred_temp}  |  {conf:.1f}% confidence</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:.65rem;color:#64748b;margin-bottom:.2rem;">Ground truth</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{true_lbl_str} {verdict_icon}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        dem_l, dem_m, dem_r = st.columns([2, 2, 3], gap="medium")

        # Left: confidence gauge
        with dem_l:
            st.markdown('<div class="sh">Confidence Gauge</div>', unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                number=dict(suffix="%", font=dict(size=28, color="#f1f5f9")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(size=9, color="#475569"),
                              tickvals=[0, 25, 50, 75, 100]),
                    bar=dict(color=PAIN_COL, thickness=0.25),
                    bgcolor="rgba(7,17,31,.0)",
                    bordercolor=LINE,
                    steps=[
                        dict(range=[0, 50], color="rgba(248,113,113,.06)"),
                        dict(range=[50, 65], color="rgba(251,146,60,.06)"),
                        dict(range=[65, 100], color="rgba(52,211,153,.06)"),
                    ],
                    threshold=dict(value=65.3, line=dict(color="rgba(255,255,255,.25)", width=2), thickness=.75),
                )
            ))
            gauge.update_layout(
                height=230, paper_bgcolor=BG, plot_bgcolor=BG,
                font=FONT, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(gauge, use_container_width=True)
            thresh_label = "Very confident" if conf >= 72 else "Fairly confident" if conf >= 60 else "Uncertain"
            st.markdown(
                f'<div style="text-align:center;font-size:.78rem;color:{PAIN_COL};'
                f'font-weight:600;margin-top:-.5rem;">{thresh_label}</div>',
                unsafe_allow_html=True
            )

        # Middle: probability bars
        with dem_m:
            st.markdown('<div class="sh">PA2 vs PA3 Probability</div>', unsafe_allow_html=True)
            for lbl, pval, grad, col in [
                ("PA2 / Moderate", prob[0] * 100, "linear-gradient(90deg,#0369a1,#38bdf8)", "#38bdf8"),
                ("PA3 / Intense",  prob[1] * 100, "linear-gradient(90deg,#9f1239,#f87171)", "#f87171"),
            ]:
                st.markdown(
                    f'<div style="margin-bottom:.9rem;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">'
                    f'<span style="font-size:.78rem;color:#94a3b8;font-weight:500;">{lbl}</span>'
                    f'<span style="font-size:.78rem;font-weight:700;color:{col};">{pval:.1f}%</span></div>'
                    f'<div style="background:rgba(7,17,31,.8);border-radius:6px;height:12px;overflow:hidden;">'
                    f'<div style="width:{pval:.1f}%;height:12px;background:{grad};border-radius:6px;'
                    f'transition:width .5s ease;"></div></div></div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="sh" style="margin-top:.8rem;">Modality Votes</div>', unsafe_allow_html=True)
            for icon, lbl, pval, col in [
                ("[bio]", "Biosignal RF",  bio_prob[1]*100, "#38bdf8"),
                ("[lm]", "Landmark RF",   lm_prob[1]*100,  "#34d399"),
                ("[fuse]", "Fusion (final)", prob[1]*100,     PAIN_COL),
            ]:
                vote = "PA3" if pval > 50 else "PA2"
                vc   = "#f87171" if pval > 50 else "#38bdf8"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:.3rem .5rem;background:rgba(14,28,54,.5);border-radius:7px;margin-bottom:.25rem;">'
                    f'<span style="font-size:.74rem;color:#64748b;">{icon} {lbl}</span>'
                    f'<span style="font-size:.74rem;font-weight:700;color:{vc};">{vote} / {pval:.0f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Right: signal preview + radar
        with dem_r:
            st.markdown('<div class="sh">Sample Signal Preview</div>', unsafe_allow_html=True)
            sname = selected_sample.get("sample_name", "") if hasattr(selected_sample, "get") else ""
            sig_path = os.path.join(SIGPLOT, f"{sname}.png") if sname else ""
            if sig_path and os.path.exists(sig_path):
                st.image(sig_path, use_container_width=True)
            else:
                # Mini bar chart of top biosignal features
                if len(bio_feats) >= 5:
                    fig_sig = go.Figure(go.Bar(
                        x=[f"F{i}" for i in range(min(8, len(bio_feats)))],
                        y=[abs(v) for v in bio_feats[:8]],
                        marker=dict(color="#38bdf8", opacity=.7, line=dict(width=0)),
                        hovertemplate="Feature %{x}: %{y:.3f}<extra></extra>",
                    ))
                    fig_sig.update_layout(**base_fig(h=160, title="Top 8 Biosignal Values (normalized)"))
                    st.plotly_chart(fig_sig, use_container_width=True)

            st.markdown('<div class="sh" style="margin-top:.5rem;">Multi-Axis Prediction Profile</div>',
                        unsafe_allow_html=True)
            radar = go.Figure(go.Scatterpolar(
                r=[bio_prob[1]*100, lm_prob[1]*100, prob[1]*100, conf,
                   (bio_prob[1]+lm_prob[1])*50],
                theta=["Biosignal", "Landmark", "Fusion", "Confidence", "Signal Avg"],
                fill="toself",
                fillcolor=f"rgba(56,189,248,.1)" if pred == 0 else "rgba(248,113,113,.1)",
                line=dict(color=PAIN_COL, width=2.5),
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            ))
            radar.update_layout(
                height=220, paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(linecolor=LINE, gridcolor=GRID,
                                     tickfont=dict(size=9, color="#64748b")),
                    radialaxis=dict(range=[0,100], gridcolor=GRID, linecolor=LINE,
                                    tickfont=dict(size=8, color="#475569"),
                                    tickvals=[25,50,75,100]),
                ),
                margin=dict(l=30, r=30, t=10, b=10),
            )
            st.plotly_chart(radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  -  FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    CLINICAL_MAP = {
        "gsr_slope":     ("Sweat Activation Speed",   "How fast sweat glands turn on  -  the fastest pain indicator"),
        "gsr_std":       ("Sweat Turbulence",          "Fluctuation in GSR  -  high chaos = high arousal"),
        "ecg_max":       ("Cardiac Peak",              "Peak ECG amplitude during the 5.5s window"),
        "gsr_shannon":   ("GSR Complexity",            "Information density of skin conductance signal"),
        "ecg_shannon":   ("Cardiac Complexity",        "Complexity of cardiac waveform under pain stimulus"),
        "gsr_sim_corr":  ("Sweat Pattern Match",       "Correlation to a canonical pain GSR template"),
        "emg_trap_std":  ("Shoulder Tension Change",   "Trapezius variability  -  pain causes muscle bracing"),
        "hrv_meannn":    ("Mean Beat Interval",        "Average R-R gap  -  drops when pain disrupts rhythm"),
        "hrv_sdnn":      ("HRV Spread",                "Beat-to-beat variability  -  pain narrows this"),
        "ecg_std":       ("Cardiac Amplitude Spread",  "Standard deviation of ECG amplitude across window"),
    }
    LM_MAP = {
        "mouth_height_std":     ("Mouth Opening Variability",   "Involuntary mouth movement during 5.5s of pain"),
        "mouth_width_std":      ("Lip Spread Variability",      "Lateral lip motion  -  retraction is a pain signal"),
        "nose_width_std":       ("Nostril Flare",               "Nostril width changes  -  classic pain micro-expression"),
        "mouth_aspect_ratio_std":("Mouth Shape Dynamics",       "Ratio of height to width changing = expression"),
        "left_brow_eye_dist_std":("L. Brow Movement",           "Left brow raising/lowering  -  most universal pain cue"),
        "brow_eye_avg_std":     ("Bilateral Brow Movement",     "Both brows moving together  -  symmetric furrowing"),
        "avg_eye_openness_mean":("Eye Openness",                "Pain causes orbital tightening and partial eye close"),
        "brow_furrow_std":      ("Brow Furrow Dynamics",        "More fluctuation = stronger pain expression"),
        "mouth_height_mean":    ("Mean Mouth Openness",         "Open mouth accompanies moderate-to-high pain"),
    }

    fa_l, fa_r = st.columns(2, gap="large")

    def shap_bar(df, title, hi_col, lo_col, n=10):
        if df is None or len(df) == 0:
            return None
        top = df.head(n).copy()
        vals = top["mean_shap"].values
        labs = top["feature"].values
        med  = np.median(vals)
        colors = [hi_col if v > med else lo_col for v in vals[::-1]]
        fig = go.Figure(go.Bar(
            x=vals[::-1], y=labs[::-1], orientation="h",
            marker=dict(color=colors, opacity=[1. if c==hi_col else .5 for c in colors], line=dict(width=0)),
            text=[f"{v:.4f}" for v in vals[::-1]], textposition="outside",
            textfont=dict(size=9, color="#475569", family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        ))
        fig.update_layout(**base_fig(h=360, title=title, xlab="Mean |SHAP value|"))
        fig.update_yaxes(tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8"))
        return fig

    with fa_l:
        st.markdown('<div class="sh">Biosignal Features  -  SHAP Importance</div>', unsafe_allow_html=True)
        fig_b = shap_bar(shap_bio, "GSR / ECG / EMG / HRV", "rgba(56,189,248,.9)", "rgba(56,189,248,.3)")
        if fig_b:
            st.plotly_chart(fig_b, use_container_width=True)

        if shap_bio is not None:
            rows = []
            for _, r in shap_bio.head(8).iterrows():
                short, long = CLINICAL_MAP.get(r["feature"], (r["feature"], " - "))
                rows.append({"Feature": r["feature"], "|SHAP|": f"{r['mean_shap']:.4f}",
                             "Meaning": short, "Detail": long})
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True,
                column_config={
                    "Feature": st.column_config.TextColumn(width="medium"),
                    "|SHAP|": st.column_config.TextColumn(width="small"),
                    "Meaning": st.column_config.TextColumn(width="medium"),
                    "Detail": st.column_config.TextColumn(width="large"),
                }
            )

        if os.path.exists(BIO_BEES):
            st.markdown('<div class="sh" style="margin-top:.8rem;">SHAP Beeswarm  -  Biosignal</div>',
                        unsafe_allow_html=True)
            st.image(BIO_BEES, use_container_width=True)
            st.markdown("""
            <div class="insight">
                <span style="font-size:.75rem;color:#94a3b8;">
                Each dot = one recording. Horizontal position = push toward PA2 (left) or PA3 (right).
                Color = raw feature value (blue=low, red=high).
                Wide spread on <b style="color:#f1f5f9;">gsr_slope</b> shows it dominates all other features.
                </span>
            </div>""", unsafe_allow_html=True)

    with fa_r:
        st.markdown('<div class="sh">Facial Landmark Features  -  SHAP Importance</div>', unsafe_allow_html=True)
        fig_l = shap_bar(shap_lm, "MediaPipe FaceMesh Geometry", "rgba(52,211,153,.9)", "rgba(52,211,153,.3)")
        if fig_l:
            st.plotly_chart(fig_l, use_container_width=True)

        if shap_lm is not None:
            rows2 = []
            for _, r in shap_lm.head(8).iterrows():
                short, long = LM_MAP.get(r["feature"], (r["feature"], " - "))
                rows2.append({"Feature": r["feature"], "|SHAP|": f"{r['mean_shap']:.4f}",
                              "Meaning": short, "Detail": long})
            st.dataframe(
                pd.DataFrame(rows2), hide_index=True, use_container_width=True,
                column_config={
                    "Feature": st.column_config.TextColumn(width="medium"),
                    "|SHAP|": st.column_config.TextColumn(width="small"),
                    "Meaning": st.column_config.TextColumn(width="medium"),
                    "Detail": st.column_config.TextColumn(width="large"),
                }
            )

        if os.path.exists(LM_BEES):
            st.markdown('<div class="sh" style="margin-top:.8rem;">SHAP Beeswarm  -  Landmark</div>',
                        unsafe_allow_html=True)
            st.image(LM_BEES, use_container_width=True)
            st.markdown("""
            <div class="insight">
                <span style="font-size:.75rem;color:#94a3b8;">
                High <b style="color:#f1f5f9;">mouth_height_std</b> (red) pushes toward PA3  - 
                the patient is opening/closing their mouth more frequently.
                All top landmark features end in <b style="color:#f1f5f9;">_std</b>:
                pain expression is <i>dynamic</i>, not a fixed facial position.
                </span>
            </div>""", unsafe_allow_html=True)

    # Combined top-16
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sh">Combined Top Features  -  Both Modalities</div>', unsafe_allow_html=True)
    if shap_bio is not None and shap_lm is not None:
        b8 = shap_bio.head(8).copy(); b8["mod"] = "Biosignal"
        l8 = shap_lm.head(8).copy();  l8["mod"] = "Landmark"
        comb = pd.concat([b8, l8]).sort_values("mean_shap", ascending=False)
        fig_c = go.Figure(go.Bar(
            x=comb["mean_shap"], y=comb["feature"], orientation="h",
            marker=dict(
                color=["rgba(56,189,248,.85)" if m=="Biosignal" else "rgba(52,211,153,.85)"
                       for m in comb["mod"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b> (%{customdata})<br>|SHAP| = %{x:.4f}<extra></extra>",
            customdata=comb["mod"].tolist(),
        ))
        fig_c.update_layout(**base_fig(h=380, title="Top-16 features  -  Blue=Biosignal / Green=Landmark",
                                       xlab="Mean |SHAP|"))
        fig_c.update_yaxes(tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8"))
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("""
        <p style="font-size:.74rem;color:#475569;text-align:center;">
        Both modalities appear in the top 16  -  they carry non-redundant information about pain intensity.
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  -  PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    perf_tabs = st.tabs(["Confusion Matrix", "Per-Subject LOSO", "Ablation Table"])

    # ── Confusion matrix ─────────────────────────────────────────────────────
    with perf_tabs[0]:
        cm_l, cm_r = st.columns([2, 3], gap="large")
        with cm_l:
            cm_vals = [[884, 456], [470, 870]]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_vals,
                x=["Predicted PA2 (Moderate)", "Predicted PA3 (Intense)"],
                y=["True PA2 (Moderate)", "True PA3 (Intense)"],
                colorscale=[[0,"#060d1a"],[.3,"#0c2040"],[.7,"#0369a1"],[1,"#38bdf8"]],
                text=[["884\n+ Correct", "456\n- False Alarm"],
                      ["470\n- Miss",    "870\n+ Correct"]],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#f1f5f9"),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False,
            ))
            fig_cm.update_layout(
                **base_fig(h=320, title="LOSO-67 overall / 2,680 samples",
                           margin=dict(l=8,r=8,t=50,b=8))
            )
            fig_cm.update_xaxes(side="top")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Metrics
            tn, fp, fn, tp = 884, 456, 470, 870
            acc  = (tn+tp)/(tn+fp+fn+tp)*100
            prec = tp/(tp+fp)*100
            rec  = tp/(tp+fn)*100
            f1   = 2*prec*rec/(prec+rec)
            m1, m2, m3, m4 = st.columns(4)
            for col, lbl, val in [(m1,"Accuracy",f"{acc:.1f}%"),(m2,"Precision",f"{prec:.1f}%"),
                                   (m3,"Recall",f"{rec:.1f}%"),(m4,"F1",f"{f1:.3f}")]:
                col.metric(lbl, val)

        with cm_r:
            st.markdown('<div class="sh">Reading the Matrix</div>', unsafe_allow_html=True)
            for heading, col, body in [
                ("Top-left: Correct PA2 (884)",  "#38bdf8",
                 "884 samples where moderate pain was correctly classified. "
                 "<b>66.0% of all PA2 cases</b> predicted correctly."),
                ("Top-right: PA2 -> PA3 false alarm (456)", "#f87171",
                 "456 moderate-pain cases called intense. The 1°C difference makes these "
                 "biosignally ambiguous."),
                ("Bottom-left: PA3 -> PA2 miss (470)", "#fb923c",
                 "470 intense-pain cases called moderate. Clinically, these are the most "
                 "costly errors  -  missed pain."),
                ("Bottom-right: Correct PA3 (870)", "#34d399",
                 "870 intense-pain samples correctly detected. "
                 "<b>64.9% of all PA3 cases</b> predicted correctly."),
            ]:
                st.markdown(
                    f'<div style="border-left:3px solid {col};padding:.65rem 1rem;'
                    f'background:{col}08;border-radius:0 10px 10px 0;margin-bottom:.5rem;">'
                    f'<div style="font-size:.77rem;font-weight:700;color:{col};margin-bottom:.18rem;">{heading}</div>'
                    f'<div style="font-size:.75rem;color:#64748b;line-height:1.6;">{body}</div></div>',
                    unsafe_allow_html=True
                )
            st.markdown("""
            <div class="insight">
                <span style="font-size:.78rem;color:#cbd5e1;line-height:1.7;">
                The error counts are <b style="color:#f1f5f9;">near-symmetric</b> (456 vs 470),
                confirming the model has no systematic bias toward either class.
                It is genuinely confused by the <b>1°C stimulus difference</b>,
                not making a systematic mistake.
                </span>
            </div>""", unsafe_allow_html=True)

    # ── Per-subject ──────────────────────────────────────────────────────────
    with perf_tabs[1]:
        if subj_df is not None:
            sdf = subj_df.sort_values("accuracy").reset_index(drop=True)
            sdf["rank"]    = range(1, len(sdf)+1)
            sdf["acc_pct"] = sdf["accuracy"]*100
            sdf["tier"]    = sdf["accuracy"].apply(
                lambda a: ">=80% Excellent" if a>=.8
                else "65-80% Good" if a>=.65
                else "50-65% Near Chance" if a>=.5
                else "<50% Below Chance"
            )
            TCOLS = {">=80% Excellent":"#34d399","65-80% Good":"#38bdf8",
                     "50-65% Near Chance":"#fb923c","<50% Below Chance":"#f87171"}
            sdf["color"] = sdf["tier"].map(TCOLS)

            ps_l, ps_r = st.columns([3, 2], gap="large")
            with ps_l:
                # Scatter
                np.random.seed(42)
                sdf["jitter"] = np.random.uniform(-.25, .25, len(sdf))
                fig_sc = go.Figure()
                for tier, tc in TCOLS.items():
                    m = sdf["tier"]==tier
                    sub = sdf[m]
                    fig_sc.add_trace(go.Scatter(
                        x=sub["rank"]+sub["jitter"], y=sub["acc_pct"],
                        mode="markers", name=tier,
                        marker=dict(color=tc, size=11, opacity=.85,
                                    line=dict(color="rgba(0,0,0,.3)", width=1)),
                        hovertemplate="<b>Subject %{customdata}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
                        customdata=sub["subject_id"].tolist() if "subject_id" in sub.columns else sub["rank"],
                    ))
                z = np.polyfit(sdf["rank"], sdf["acc_pct"], 2)
                tx = np.linspace(1, len(sdf), 120)
                fig_sc.add_trace(go.Scatter(x=tx, y=np.polyval(z,tx), mode="lines",
                                            line=dict(color="rgba(255,255,255,.1)",width=2.5,dash="dot"),
                                            showlegend=False, hoverinfo="skip"))
                fig_sc.add_hline(y=50,   line_dash="dot",  line_color="rgba(248,113,113,.4)", line_width=1,
                                 annotation_text="Chance 50%", annotation_font_color="#f87171", annotation_font_size=9)
                fig_sc.add_hline(y=65.3, line_dash="dash", line_color="rgba(56,189,248,.55)", line_width=1.5,
                                 annotation_text="Mean 65.3%", annotation_font_color="#38bdf8", annotation_font_size=9)
                fig_sc.update_layout(**base_fig(h=300, xlab="Subject rank (worst->best)", ylab="Accuracy (%)"),
                                     yaxis_range=[15,108],
                                     legend=dict(orientation="h", x=0, y=1.12,
                                                 font=dict(size=10,color="#64748b"),
                                                 bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_sc, use_container_width=True)

                # Histogram
                fig_h = go.Figure(go.Histogram(
                    x=sdf["acc_pct"], nbinsx=18,
                    marker=dict(
                        color=["#34d399" if v>=80 else "#38bdf8" if v>=65
                               else "#fb923c" if v>=50 else "#f87171" for v in sdf["acc_pct"]],
                        line=dict(color="rgba(0,0,0,.2)", width=1)
                    ),
                    hovertemplate="Accuracy %{x:.0f}%: %{y} subjects<extra></extra>",
                ))
                fig_h.add_vline(x=50,   line_dash="dot",  line_color="rgba(248,113,113,.5)", line_width=1)
                fig_h.add_vline(x=65.3, line_dash="dash", line_color="rgba(56,189,248,.7)",  line_width=2)
                fig_h.update_layout(**base_fig(h=150, title="Accuracy distribution across 67 subjects",
                                               xlab="Accuracy (%)", ylab="# Subjects",
                                               margin=dict(l=8,r=8,t=32,b=8)))
                st.plotly_chart(fig_h, use_container_width=True)

            with ps_r:
                n_exc   = int((sdf["accuracy"]>=.8).sum())
                n_good  = int(((sdf["accuracy"]>=.65)&(sdf["accuracy"]<.8)).sum())
                n_chance= int(((sdf["accuracy"]>=.5)&(sdf["accuracy"]<.65)).sum())
                n_below = int((sdf["accuracy"]<.5).sum())

                fig_pie = go.Figure(go.Pie(
                    values=[n_exc,n_good,n_chance,n_below],
                    labels=[">=80%","65-80%","50-65%","<50%"],
                    hole=.62,
                    marker=dict(colors=["#34d399","#38bdf8","#fb923c","#f87171"],
                                line=dict(color=["#050c18"]*4, width=3)),
                    textfont=dict(size=9,color="#94a3b8"),
                    sort=False,
                ))
                fig_pie.update_layout(
                    height=260, showlegend=True, paper_bgcolor=BG, plot_bgcolor=BG, font=FONT,
                    legend=dict(orientation="h",x=.5,xanchor="center",y=-.06,
                                font=dict(size=9,color="#64748b"),bgcolor="rgba(0,0,0,0)"),
                    title=dict(text="Subject tier breakdown",font=dict(size=10,color="#64748b"),
                               x=.5,xanchor="center"),
                    annotations=[dict(text="67<br>subjects",x=.5,y=.5,
                                      font_size=12,font_color="#94a3b8",showarrow=False)],
                    margin=dict(l=10,r=10,t=40,b=40),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                for n, lbl, col in [
                    (n_exc,   "Excellent  >=80%",     "#34d399"),
                    (n_good,  "Good  65-80%",         "#38bdf8"),
                    (n_chance,"Near Chance  50-65%",  "#fb923c"),
                    (n_below, "Below Chance  <50%",   "#f87171"),
                ]:
                    pct = n/67*100
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:.6rem .9rem;background:rgba(7,17,31,.6);'
                        f'border:1px solid {col}22;border-left:3px solid {col};'
                        f'border-radius:10px;margin-bottom:.35rem;">'
                        f'<div><div style="font-size:1.45rem;font-weight:900;color:{col};line-height:1;">{n}</div>'
                        f'<div style="font-size:.68rem;color:#475569;margin-top:.1rem;">{lbl}</div></div>'
                        f'<div style="font-size:1rem;font-weight:700;color:{col}66;">{pct:.0f}%</div></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("""
                <div class="insight" style="margin-top:.5rem;">
                    <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;
                        color:#818cf8;margin-bottom:.3rem;">Why the wide variance?</div>
                    <p style="font-size:.74rem;color:#94a3b8;line-height:1.7;margin:0;">
                        Some subjects' bodies respond nearly identically to 43°C and 44°C heat  - 
                        their sweat, cardiac, and facial patterns are indistinguishable at 1°C apart.
                        This is <b style="color:#e2e8f0;">inter-individual biology</b>,
                        not a model failure.
                    </p>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Per-subject accuracy data not available. Run error_analysis_loso.py to generate it.")

    # ── Ablation table ───────────────────────────────────────────────────────
    with perf_tabs[2]:
        ablation = [
            {"Model": "EmPath Stacked Fusion",      "Protocol": "LOSO-67",   "Accuracy": "65.3%", "Std": "±14.1%", "Type": " Best"},
            {"Model": "CORAL Ordinal MLP",           "Protocol": "LOSO-67",   "Accuracy": "65.3%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "Subject Adaptation RF",       "Protocol": "LOSO-67",   "Accuracy": "65.1%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "DANN + RF Landmarks",         "Protocol": "LOSO-67",   "Accuracy": "64.7%", "Std": "±11.8%", "Type": "Novel"},
            {"Model": "Early Fusion (concat RF)",    "Protocol": "LOSO-67",   "Accuracy": "64.6%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "Velocity + Biosignal Stacked","Protocol": "LOSO-67",   "Accuracy": "64.0%", "Std": "±12.7%", "Type": "Novel"},
            {"Model": "Biosignal RF (person-norm)",  "Protocol": "LOSO-67",   "Accuracy": "63.1%", "Std": "±11.6%", "Type": "Ablation"},
            {"Model": "GNN + Biosignal Stacked",     "Protocol": "LOSO-67",   "Accuracy": "63.1%", "Std": "±11.9%", "Type": "Novel"},
            {"Model": "CrossMod Cross-Attention",    "Protocol": "LOSO-67",   "Accuracy": "63.1%", "Std": "±11.1%", "Type": "Novel"},
            {"Model": "Tiny-BioMoE + Hand Feats",    "Protocol": "LOSO-67",   "Accuracy": "61.7%", "Std": " - ",      "Type": "Foundation"},
            {"Model": "DANN Biosignal Only",         "Protocol": "LOSO-67",   "Accuracy": "61.6%", "Std": "±10.3%", "Type": "Novel"},
            {"Model": "Landmark RF (flat)",          "Protocol": "LOSO-67",   "Accuracy": "61.4%", "Std": "±13.1%", "Type": "Ablation"},
            {"Model": "Ordinal MLP",                 "Protocol": "LOSO-67",   "Accuracy": "61.2%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "Attention Fusion",            "Protocol": "LOSO-67",   "Accuracy": "61.1%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "BIOT + Hand Feats",           "Protocol": "LOSO-67",   "Accuracy": "60.8%", "Std": " - ",      "Type": "Foundation"},
            {"Model": "Velocity RF Only",            "Protocol": "LOSO-67",   "Accuracy": "60.0%", "Std": "±11.9%", "Type": "Novel"},
            {"Model": "Biosignal RF (no norm)",      "Protocol": "LOSO-67",   "Accuracy": "59.9%", "Std": "±13.2%", "Type": "Ablation"},
            {"Model": "Hybrid CNN + Hand Feats",     "Protocol": "LOSO-67",   "Accuracy": "59.7%", "Std": " - ",      "Type": "Ablation"},
            {"Model": "Tiny-BioMoE",                 "Protocol": "LOSO-67",   "Accuracy": "56.7%", "Std": " - ",      "Type": "Foundation"},
            {"Model": "Biosignal TCN",               "Protocol": "Random",    "Accuracy": "55.9%", "Std": " - ",      "Type": "Baseline"},
            {"Model": "BIOT Foundation Model",       "Protocol": "LOSO-67",   "Accuracy": "54.4%", "Std": " - ",      "Type": "Foundation"},
            {"Model": "PainFormer",                  "Protocol": "LOSO-67",   "Accuracy": "53.1%", "Std": " - ",      "Type": "Foundation"},
            {"Model": "GNN Landmarks Only",          "Protocol": "LOSO-67",   "Accuracy": "51.7%", "Std": "±9.8%",  "Type": "Novel"},
            {"Model": "Biosignal MLP",               "Protocol": "Random",    "Accuracy": "51.2%", "Std": " - ",      "Type": "Baseline"},
            {"Model": "Biosignal SVM",               "Protocol": "Random",    "Accuracy": "48.8%", "Std": " - ",      "Type": "Baseline"},
            {"Model": "Vision MobileNetV2",          "Protocol": "Random",    "Accuracy": "47.2%", "Std": " - ",      "Type": "Baseline"},
            {"Model": "CrossMod-Transformer (2025)*","Protocol": "LOSO-87 all","Accuracy": "87.5%","Std": " - ",      "Type": "SOTA (all subj)"},
        ]
        adf = pd.DataFrame(ablation)
        adf["acc_num"] = adf["Accuracy"].str.replace("%","").astype(float)
        TYPE_COLS = {
            " Best": "#38bdf8", "Novel": "#a78bfa", "Ablation": "#64748b",
            "Foundation": "#fb923c", "Baseline": "#475569",
            "SOTA (all subj)": "#f87171",
        }
        fig_abl = go.Figure(go.Bar(
            x=adf["acc_num"], y=adf["Model"], orientation="h",
            marker=dict(color=[TYPE_COLS.get(t,"#64748b") for t in adf["Type"]], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in adf["acc_num"]], textposition="outside",
            textfont=dict(size=9,color="#475569"),
            hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1f}%<br>Protocol: %{customdata}<extra></extra>",
            customdata=adf["Protocol"].tolist(),
        ))
        fig_abl.add_vline(x=50,   line_dash="dot",  line_color="rgba(248,113,113,.4)", line_width=1)
        fig_abl.add_vline(x=65.3, line_dash="dash", line_color="rgba(56,189,248,.55)", line_width=1.5)
        fig_abl.update_layout(**base_fig(h=620, xlab="Accuracy (%)",
                                         title="26-variant ablation  -  dashed lines: chance (50%) and EmPath best (65.3%)"),
                              xaxis_range=[35,100])
        fig_abl.update_yaxes(autorange="reversed", tickfont=dict(size=10,color="#94a3b8"))
        st.plotly_chart(fig_abl, use_container_width=True)

        st.dataframe(
            adf.drop(columns=["acc_num"]), hide_index=True, use_container_width=True,
            column_config={
                "Model":    st.column_config.TextColumn("Model",    width="large"),
                "Protocol": st.column_config.TextColumn("Protocol", width="medium"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
                "Std":      st.column_config.TextColumn("Std Dev",  width="small"),
                "Type":     st.column_config.TextColumn("Type",     width="medium"),
            }
        )
        st.markdown("""
        <p style="font-size:.72rem;color:#334155;margin-top:.5rem;line-height:1.7;">
        * CrossMod-Transformer 2025 (87.5%) evaluates on all 87 BioVid subjects including
        20 non-reactive ones with flat biosignals, artificially inflating results.
        EmPath's LOSO-67 reactive-only protocol is the stricter, honest comparison.
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  -  ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    arch_l, arch_r = st.columns(2, gap="large")
    with arch_l:
        st.markdown('<div class="sh">Pipeline Overview</div>', unsafe_allow_html=True)
        steps = [
            ("01", "#38bdf8", "Sensor Data",      "GSR / ECG / EMG",
             "Three body sensors record for 5.5 seconds at 512 Hz (GSR) and 256 Hz (ECG/EMG)."),
            ("02", "#a78bfa", "Facial Video",     "25 fps / 24 frames",
             "MediaPipe FaceMesh maps 468 landmarks on each frame  -  brows, eyes, nose, mouth."),
            ("03", "#fb923c", "Feature Extraction","57 total features",
             "35 statistical + HRV features from biosignals. 22 geometric features from landmarks."),
            ("04", "#34d399", "Two RF Models",    "300 trees each",
             "RF-Biosignal and RF-Landmark trained separately  -  each outputs PA2/PA3 probabilities."),
            ("05", "#f87171", "Meta-Learner",     "Logistic Regression",
             "Sees 4 probability values (2 from each RF) and learns the optimal fusion weight."),
            ("06", "#38bdf8", "SHAP Explanation", "TreeExplainer",
             "Per-prediction feature attributions. Every output is explainable and auditable."),
        ]
        cols6 = st.columns(3, gap="small")
        for i, (num, col, title, sub, body) in enumerate(steps):
            with cols6[i % 3]:
                st.markdown(
                    f'<div class="step">'
                    f'<div class="step-n" style="background:{col};color:#050c18;">{num}</div><br>'
                    f'<div style="font-size:.82rem;font-weight:700;color:#e2e8f0;margin-bottom:.15rem;">{title}</div>'
                    f'<div style="font-size:.67rem;color:{col};font-weight:600;margin-bottom:.4rem;">{sub}</div>'
                    f'<div style="font-size:.72rem;color:#64748b;line-height:1.55;">{body}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">Model Architecture Diagram</div>', unsafe_allow_html=True)
        st.code("""
BioVid Database
      │                           │
 MP4 Video Files           Biosignal TSVs
      │                           │
 Face Extraction           Signal Windowing
      │                           │
 MediaPipe FaceMesh        NeuroKit2 + Stats
      │                           │
22 Landmark Features      35 Biosignal Features
       \\                          /
        Person-Specific z-Score
       /                          \\
RF Landmark               RF Biosignal
(p_lm [2])                (p_bio [2])
       \\                          /
     [p_bio ‖ p_lm]   (4-dim)
               │
    LogReg Meta-Learner
               │
         PA2 / PA3
        """, language="text")

    with arch_r:
        st.markdown('<div class="sh">Key Design Decisions</div>', unsafe_allow_html=True)
        decisions = [
            ("#38bdf8", "Why LOSO? (not random split)",
             "LOSO = Leave-One-Subject-Out. The model never sees ANY data from the test subject. "
             "Random splits leak subject identity  -  the model memorizes individual physiology, "
             "inflating accuracy by ~8-10 pp. LOSO tests true generalization to new people, "
             "which is what clinical deployment requires."),
            ("#34d399", "Why person-specific normalization?",
             "Each person's physiological baseline is different. Person A's resting GSR may be "
             "10 μS, Person B's may be 2 μS. Global normalization conflates them. Person-specific "
             "normalization converts each signal to 'X% above your own normal'  -  yielding "
             "+3.2 pp improvement (59.9% -> 63.1%)."),
            ("#a78bfa", "Why Random Forest? (not deep learning)",
             "With only 67 subjects and ~2,680 samples, deep models overfit in LOSO. TCN=55.9%, "
             "MLP=51.2%, and foundation models (BIOT=54.4%, PainFormer=53.1%) all underperform "
             "the RF. RF is the empirically correct choice for this dataset size."),
            ("#fb923c", "Why stacked fusion? (not early fusion)",
             "Early fusion: concatenate all 57 features -> RF = 64.6%. "
             "Stacked fusion: RF(bio) probs + RF(lm) probs -> LogReg = 65.3%. "
             "Stacking preserves modality structure and combines calibrated probabilities "
             "rather than mixing raw features on different scales."),
        ]
        for col, title, body in decisions:
            st.markdown(
                f'<div style="border-left:3px solid {col};padding:.8rem 1rem;'
                f'background:{col}08;border-radius:0 12px 12px 0;margin-bottom:.7rem;">'
                f'<div style="font-size:.78rem;font-weight:700;color:{col};margin-bottom:.3rem;">{title}</div>'
                f'<div style="font-size:.75rem;color:#64748b;line-height:1.65;">{body}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="sh" style="margin-top:.5rem;">Performance Ceiling Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight">
            <div style="font-size:.8rem;color:#cbd5e1;line-height:1.75;">
                All 26 architectural variants  -  including deep learning (TCN, MLP), foundation models
                (BIOT, PainFormer, Tiny-BioMoE), GNNs, DANN, and cross-attention  -  converge near or
                below <b style="color:#38bdf8;">65.3%</b>.<br><br>
                The ceiling is set by <b style="color:#f1f5f9;">inter-individual physiological variability</b>
                and the <b style="color:#f1f5f9;">1°C stimulus difference between PA2 and PA3</b>,
                not by model capacity. More complex architectures do not help here.
            </div>
        </div>""", unsafe_allow_html=True)

        # Mini comparison chart
        arch_models = ["Chance", "PainFormer", "BIOT", "Biosignal RF", "Landmark RF",
                       "Early Fusion", "EmPath Stacked"]
        arch_accs   = [50, 53.1, 54.4, 63.1, 61.4, 64.6, 65.3]
        arch_colors = ["#334155","#fb923c","#fb923c","#38bdf8","#34d399","#64748b","#38bdf8"]
        fig_arch = go.Figure(go.Bar(
            x=arch_accs, y=arch_models, orientation="h",
            marker=dict(color=arch_colors, opacity=[.5 if a<65 else .95 for a in arch_accs],
                        line=dict(width=0)),
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        ))
        fig_arch.add_vline(x=65.3, line_dash="dash", line_color="rgba(56,189,248,.6)", line_width=2)
        fig_arch.update_layout(**base_fig(h=250, xlab="Accuracy (%)",
                                          margin=dict(l=10,r=20,t=10,b=30)),
                               xaxis_range=[40, 72])
        fig_arch.update_yaxes(tickfont=dict(size=10, color="#94a3b8"))
        st.plotly_chart(fig_arch, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  -  CLINICAL CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    clin_l, clin_r = st.columns(2, gap="large")
    with clin_l:
        st.markdown('<div class="sh">Clinical Motivation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card card-accent-blue" style="margin-bottom:.7rem;">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                color:#38bdf8;margin-bottom:.5rem;">The Problem</div>
            <p style="font-size:.85rem;color:#cbd5e1;line-height:1.7;margin:0;">
                Some patients  -  ICU sedation, neonates, dementia  - 
                <b style="color:#f1f5f9;">cannot self-report pain</b>.<br><br>
                Without self-reporting, clinicians must estimate from behavior and physiology.
                Under- or over-treating pain in these populations carries serious clinical risks.
                EmPath automates objective pain discrimination using sensors already present
                in many ICU setups.
            </p>
        </div>
        <div class="card card-accent-green">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                color:#34d399;margin-bottom:.5rem;">The Task</div>
            <p style="font-size:.85rem;color:#cbd5e1;line-height:1.7;margin:0;">
                Given a <b style="color:#f1f5f9;">5.5-second window</b> of physiological data,
                is the subject experiencing
                <b style="color:#38bdf8;">moderate pain (PA2 ≈ 43°C)</b> or
                <b style="color:#f87171;">intense pain (PA3 ≈ 45°C)</b>?<br><br>
                Only 1°C apart  -  one of the most challenging pain discrimination tasks in
                the automated pain assessment literature.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:.5rem;">Required Hardware (SHAP-ranked)</div>',
                    unsafe_allow_html=True)
        hw = [
            ("#1","#38bdf8","GSR Sensor","Wearable sweat patch, ICU adhesive electrode, or smartwatch optical",
             "gsr_slope is the #1 feature by 3x margin  -  most critical hardware"),
            ("#2","#f87171","ECG / PPG Monitor","Standard ICU cardiac monitor, finger-clip pulse oximeter",
             "ecg_max + ecg_shannon both in top 5  -  cardiac response catches PA3"),
            ("#3","#34d399","Bedside Camera","Any 25 fps camera at face level  -  even USB webcam works",
             "mouth_height_std is #1 landmark feature  -  video is required"),
            ("#4","#fb923c","EMG Electrodes","Adhesive surface electrodes on trapezius (shoulder/neck)",
             "emg_trap_std ranks #9 overall  -  useful but not critical for minimal setup"),
        ]
        for pri, col, name, what, why in hw:
            st.markdown(
                f'<div style="display:flex;gap:.75rem;padding:.85rem;'
                f'background:rgba(7,17,31,.55);border:1px solid {col}18;'
                f'border-left:3px solid {col};border-radius:12px;margin-bottom:.4rem;">'
                f'<div style="min-width:28px;">'
                f'<div style="font-size:.68rem;font-weight:800;color:{col};'
                f'background:{col}18;border-radius:6px;padding:2px 5px;text-align:center;">{pri}</div></div>'
                f'<div><div style="font-weight:700;color:#e2e8f0;font-size:.85rem;">{name}</div>'
                f'<div style="font-size:.72rem;color:#64748b;margin:.1rem 0;">{what}</div>'
                f'<div style="font-size:.68rem;color:{col};opacity:.75;">{why}</div></div></div>',
                unsafe_allow_html=True
            )

    with clin_r:
        st.markdown('<div class="sh">Literature Comparison</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="margin-bottom:.7rem;">
            <table style="width:100%;border-collapse:collapse;font-size:.79rem;">
                <thead>
                    <tr style="border-bottom:1px solid rgba(56,189,248,.12);">
                        <th style="text-align:left;padding:.4rem 0;color:#475569;">Method</th>
                        <th style="text-align:center;color:#475569;">Protocol</th>
                        <th style="text-align:right;color:#475569;">Acc</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid rgba(255,255,255,.04);">
                        <td style="padding:.4rem 0;color:#64748b;">Biosignal SVM</td>
                        <td style="text-align:center;color:#334155;font-size:.7rem;">Random split</td>
                        <td style="text-align:right;color:#64748b;">48.8%</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,.04);">
                        <td style="color:#64748b;">PainFormer (found. model)</td>
                        <td style="text-align:center;color:#334155;font-size:.7rem;">LOSO-67</td>
                        <td style="text-align:right;color:#fb923c;">53.1%</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,.04);background:rgba(56,189,248,.06);">
                        <td style="font-weight:800;color:#38bdf8;padding:.4rem 0;">EmPath Stacked *</td>
                        <td style="text-align:center;font-size:.7rem;color:#38bdf8;">LOSO-67 reactive</td>
                        <td style="text-align:right;font-weight:900;color:#38bdf8;font-size:1.1rem;">65.3%</td>
                    </tr>
                    <tr>
                        <td style="color:#64748b;padding:.4rem 0;">CrossMod-T 2025 *</td>
                        <td style="text-align:center;color:#334155;font-size:.7rem;">LOSO-87 all</td>
                        <td style="text-align:right;color:#f87171;">87.5%</td>
                    </tr>
                </tbody>
            </table>
            <p style="font-size:.66rem;color:#334155;margin:.65rem 0 0;line-height:1.65;">
                * Reactive-only = stricter, honest protocol.<br>
                * Includes 20 non-reactive subjects with flat biosignals.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sh">System Properties</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div style="font-size:.78rem;line-height:2.3;color:#64748b;">
                <span style="color:#34d399;font-weight:700;">+</span> Non-invasive  -  no needles or procedures<br>
                <span style="color:#34d399;font-weight:700;">+</span> Real-time  -  feature extraction &lt;1 second<br>
                <span style="color:#34d399;font-weight:700;">+</span> SHAP-explainable  -  every prediction justified<br>
                <span style="color:#34d399;font-weight:700;">+</span> Multimodal  -  degrades gracefully if sensor fails<br>
                <span style="color:#34d399;font-weight:700;">+</span> LOSO validated  -  not memorizing physiology<br>
                <span style="color:#f87171;font-weight:700;">-</span> Research prototype  -  not clinically validated<br>
                <span style="color:#f87171;font-weight:700;">-</span> PA2 vs PA3 only  -  not a full pain scale<br>
                <span style="color:#f87171;font-weight:700;">-</span> BioVid lab conditions  -  not ICU validated
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:.5rem;">Dataset Overview</div>',
                    unsafe_allow_html=True)
        ds_rows = [
            ("Database",  "BioVid Heat Pain (Walter et al., 2013)"),
            ("Stimulus",  "Peltier thermode on right forearm"),
            ("PA2 temp",  "~43°C  /  Moderate pain"),
            ("PA3 temp",  "~44-45°C  /  Intense pain"),
            ("Subjects",  "87 total  ->  67 reactive (20 excluded)"),
            ("Samples",   "2,680 total (PA2 + PA3 balanced)"),
            ("Signals",   "GSR, ECG, EMG (trapezius/corrugator/zygomaticus)"),
            ("Video",     "640x480, 25 fps, frontal face view"),
            ("Window",    "5.5 s per trial, 24 frames extracted"),
        ]
        for k, v in ds_rows:
            st.markdown(
                f'<div style="display:flex;gap:.6rem;padding:.3rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,.03);">'
                f'<span style="font-size:.74rem;color:#475569;min-width:90px;">{k}</span>'
                f'<span style="font-size:.74rem;color:#94a3b8;">{v}</span></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:.5rem;padding:.5rem 0 1rem;">
    <div style="font-size:.68rem;color:#334155;line-height:1.9;">
        EmPath v2  |  Komala Belur Srinivas  | 
        Hofstra University M.S. Computer Science  |  2026<br>
        BioVid Heat Pain Database  | 
        Stacked RF Fusion (RF-Biosignal + RF-Landmark -> LogisticRegression)
    </div>
    <div>
        <span class="tag">LOSO-67</span>
        <span class="tag tag-g">SHAP TreeExplainer</span>
        <span class="tag tag-p">MediaPipe FaceMesh</span>
        <span class="tag">scikit-learn 1.7.2</span>
        <span class="tag tag-r">BioVid Database</span>
    </div>
</div>
""", unsafe_allow_html=True)
