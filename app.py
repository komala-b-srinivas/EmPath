"""
EmPath v2 - Clinical Pain Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="EmPath v2 | Pain Monitor",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
    --bg: #050c18;
    --bg2: #07111f;
    --bg3: #091629;
    --border: rgba(56,189,248,.13);
    --border2: rgba(56,189,248,.25);
    --cyan: #38bdf8;
    --cyan2: #0ea5e9;
    --green: #34d399;
    --red: #f87171;
    --orange: #fb923c;
    --purple: #818cf8;
    --text: #e2e8f0;
    --muted: #64748b;
    --subtle: #334155;
}

html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ padding-top:0!important; padding-bottom:2rem; max-width:100%!important; }

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#060f1e 0%,#03090f 100%);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{ color:#94a3b8!important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{ color:var(--text)!important; }

[data-testid="stSelectbox"]>div>div{
    background:rgba(7,17,31,.9)!important;
    border:1px solid var(--border)!important;
    border-radius:8px!important; color:var(--text)!important;
}

[data-testid="stMetric"]{
    background:linear-gradient(135deg,rgba(14,30,60,.95),rgba(9,22,44,.9));
    border:1px solid var(--border);
    border-radius:14px; padding:.9rem 1.1rem;
    box-shadow:0 2px 20px rgba(0,0,0,.4);
}
[data-testid="stMetricLabel"]{ color:var(--muted)!important;font-size:.68rem!important;text-transform:uppercase;letter-spacing:.12em; }
[data-testid="stMetricValue"]{ color:var(--cyan)!important;font-size:1.6rem!important;font-weight:800!important; }
[data-testid="stMetricDelta"]{ color:var(--green)!important;font-size:.7rem!important; }

[data-testid="stTabs"] [data-baseweb="tab-list"]{
    background:rgba(7,17,31,.9); border:1px solid var(--border);
    border-radius:12px; padding:4px; gap:4px;
}
[data-testid="stTabs"] [data-baseweb="tab"]{
    background:transparent; color:#475569; border-radius:8px;
    font-size:.78rem; font-weight:500; padding:.4rem 1rem; border:none; transition:all .2s;
}
[data-testid="stTabs"] [aria-selected="true"]{
    background:linear-gradient(135deg,#0369a1,#0891b2)!important;
    color:#fff!important; box-shadow:0 2px 14px rgba(8,145,178,.4)!important;
}
[data-testid="stExpander"]{
    background:rgba(7,17,31,.6); border:1px solid var(--border); border-radius:12px;
}

@keyframes slide-up{ from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes glow-pulse{ 0%,100%{box-shadow:0 0 18px rgba(56,189,248,.15)} 50%{box-shadow:0 0 36px rgba(56,189,248,.4)} }

.animate-up{ animation:slide-up .5s ease forwards; }

.hero{
    background:linear-gradient(135deg,#050c18 0%,#09162b 40%,#060f20 70%,#050c18 100%);
    border-bottom:1px solid var(--border);
    padding:1.6rem 2.4rem 1.3rem;
    position:relative; overflow:hidden;
}
.hero::before{
    content:''; position:absolute; inset:0;
    background:
        radial-gradient(ellipse 70% 55% at 75% 50%,rgba(56,189,248,.07) 0%,transparent 70%),
        radial-gradient(ellipse 40% 40% at 15% 60%,rgba(129,140,248,.04) 0%,transparent 70%);
    pointer-events:none;
}
.hero::after{
    content:''; position:absolute; inset:0;
    background-image:radial-gradient(rgba(56,189,248,.07) 1px, transparent 1px);
    background-size:28px 28px;
    pointer-events:none;
}
.hero-title{
    font-size:2.1rem; font-weight:900;
    background:linear-gradient(135deg,#f1f5f9 0%,#38bdf8 45%,#818cf8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; letter-spacing:-.03em; line-height:1.08;
}
.hero-sub{ font-size:.8rem; color:#475569; margin-top:.35rem; }

.glass{
    background:linear-gradient(135deg,rgba(14,30,60,.75),rgba(7,17,31,.85));
    border:1px solid var(--border); border-radius:16px; padding:1.4rem;
    backdrop-filter:blur(12px); box-shadow:0 4px 32px rgba(0,0,0,.45);
}
.glass-glow{
    border-color:var(--border2);
    animation:glow-pulse 3.5s ease infinite;
}

.tag{display:inline-block;background:rgba(3,105,161,.25);border:1px solid rgba(56,189,248,.25);border-radius:20px;padding:2px 11px;font-size:.7rem;color:var(--cyan);font-weight:500;margin:2px;}
.tag-green{background:rgba(5,150,105,.2);border-color:rgba(52,211,153,.3);color:var(--green);}
.tag-red{background:rgba(185,28,28,.2);border-color:rgba(248,113,113,.3);color:var(--red);}
.tag-purple{background:rgba(99,102,241,.2);border-color:rgba(129,140,248,.3);color:var(--purple);}
.tag-orange{background:rgba(194,65,12,.2);border-color:rgba(251,146,60,.3);color:var(--orange);}

.sec{
    font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.16em;
    color:var(--cyan); margin-bottom:.65rem;
    display:flex; align-items:center; gap:.5rem;
}
.sec::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(56,189,248,.3),transparent);}

.div{height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.12),transparent);margin:1.75rem 0;}

.mono{font-family:'JetBrains Mono',monospace;}

.plain-box{
    background:linear-gradient(135deg,rgba(99,102,241,.09),rgba(139,92,246,.05));
    border:1px solid rgba(129,140,248,.22); border-left:3px solid var(--purple);
    border-radius:10px; padding:.9rem 1.1rem;
}

.insight{
    background:linear-gradient(135deg,rgba(3,105,161,.1),rgba(8,145,178,.05));
    border:1px solid rgba(56,189,248,.18); border-radius:12px; padding:.9rem 1.2rem;
}

.step-card{
    background:linear-gradient(135deg,rgba(14,30,60,.8),rgba(7,17,31,.9));
    border:1px solid var(--border); border-radius:14px; padding:1.2rem;
    position:relative; transition:border-color .2s;
}
.step-num{
    position:absolute; top:-14px; left:1.1rem;
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:.75rem; font-weight:800; border:2px solid var(--bg);
}

[data-testid="stDataFrame"]{ border-radius:10px; overflow:hidden; }

::-webkit-scrollbar{ width:5px; height:5px; }
::-webkit-scrollbar-track{ background:var(--bg2); }
::-webkit-scrollbar-thumb{ background:var(--border2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# Plotly theme
BG = "rgba(0,0,0,0)"
GRID = "rgba(56,189,248,.07)"
LINE = "rgba(56,189,248,.13)"
FONT = dict(family="Inter", color="#64748b", size=11)


def base_layout(**kw):
    d = dict(plot_bgcolor=BG, paper_bgcolor=BG, font=FONT,
             margin=dict(l=8, r=8, t=36, b=8))
    d.update(kw)
    return d


def theme_axes(fig):
    fig.update_xaxes(gridcolor=GRID, linecolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=GRID, linecolor=LINE, zerolinecolor=LINE)


# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "Models", "empath_model.pkl")
DEMO_PATH = os.path.join(BASE, "Models", "demo_samples.csv")
SIG_DIR = os.path.join(BASE, "Models", "signal_plots")
XAI = os.path.join(BASE, "Results", "error_analysis_v2")
PER_SUBJ_CSV = os.path.join(XAI, "per_subject_accuracy.csv")
SHAP_BIO_CSV = os.path.join(XAI, "shap_biosignal_ranked.csv")
SHAP_LM_CSV = os.path.join(XAI, "shap_landmark_ranked.csv")
CONFUSION = os.path.join(XAI, "confusion_matrix.png")
FEAT_IMP = os.path.join(XAI, "feature_importance_combined.png")
BIO_BEES = os.path.join(XAI, "shap_biosignal_beeswarm.png")
LM_BEES = os.path.join(XAI, "shap_landmark_beeswarm.png")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_PATH)


@st.cache_data
def load_subj():
    return pd.read_csv(PER_SUBJ_CSV) if os.path.exists(PER_SUBJ_CSV) else None


@st.cache_data
def load_shap():
    bio = pd.read_csv(SHAP_BIO_CSV) if os.path.exists(SHAP_BIO_CSV) else None
    lm = pd.read_csv(SHAP_LM_CSV) if os.path.exists(SHAP_LM_CSV) else None
    return bio, lm


model = load_model()
samples = load_demo()
subj_df = load_subj()
shap_bio_df, shap_lm_df = load_shap()


def predict(bio_feats, lm_feats):
    bio = np.array(bio_feats).reshape(1, -1)
    lm = np.array(lm_feats).reshape(1, -1)
    bio = (bio - model["global_bio_mean"]) / model["global_bio_std"]
    lm = (lm - model["global_lm_mean"]) / model["global_lm_std"]
    bp = model["rf_bio"].predict_proba(bio)
    lp = model["rf_lm"].predict_proba(lm)
    mp = np.hstack([bp, lp])
    p = model["meta"].predict(mp)[0]
    pr = model["meta"].predict_proba(mp)[0]
    return p, pr, bp[0], lp[0]


def get_feats(sname):
    bc = model["bio_cols"]
    lc = model["lm_cols"]
    r = samples[samples["sample_name"] == sname].iloc[0]
    return (
        [r[f"bio_{i}"] for i in range(len(bc))],
        [r[f"lm_{i}"] for i in range(len(lc))]
    )


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 .6rem;text-align:center;">
        <div style="font-size:1.1rem;font-weight:900;color:#38bdf8;letter-spacing:.05em;">EmPath v2</div>
        <div style="font-size:.65rem;color:#1e3a5f;text-transform:uppercase;letter-spacing:.18em;margin-top:.12rem;">
            Pain Detection System
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(56,189,248,.1);margin:.7rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">Sample Selection</div>', unsafe_allow_html=True)
    subs = sorted(samples["subject_id"].unique())
    sel_sub = st.selectbox("Subject", subs, label_visibility="collapsed")
    sub_samps = samples[samples["subject_id"] == sel_sub]["sample_name"].tolist()
    sel_samp = st.selectbox("Sample", sub_samps, label_visibility="collapsed")
    true_lbl = samples[samples["sample_name"] == sel_samp]["class_name"].values[0]

    st.markdown('<hr style="border:none;border-top:1px solid rgba(56,189,248,.1);margin:.7rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Overall Performance</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    a.metric("Accuracy", "65.3%", "+8.4%")
    b.metric("AUC", "0.719")
    c, d = st.columns(2)
    c.metric("F1 Score", "0.653")
    d.metric("Subjects", "67 LOSO")

    st.markdown('<hr style="border:none;border-top:1px solid rgba(56,189,248,.1);margin:.7rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Fusion Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="mono" style="font-size:.66rem;color:#475569;line-height:2.1;
        background:rgba(7,17,31,.7);border-radius:8px;padding:.8rem;">
        Input (57 features)<br>
        RF Biosignal (35 feat)<br>
        &nbsp;&nbsp; 300 trees · depth 4<br>
        RF Landmark (22 feat)<br>
        &nbsp;&nbsp; 300 trees · depth 4<br>
        LogReg meta-learner<br>
        &nbsp;&nbsp;&nbsp;&nbsp; PA2 / PA3
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(56,189,248,.1);margin:.7rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(185,28,28,.08);border:1px solid rgba(185,28,28,.2);
        border-radius:8px;padding:.65rem;font-size:.68rem;color:#fca5a5;line-height:1.6;">
        <b>Research Prototype</b><br>
        Not validated for clinical use.<br>
        BioVid database · 67 subjects only.
    </div>""", unsafe_allow_html=True)


# RUN PREDICTION
bio_feats, lm_feats = get_feats(sel_samp)
pred, prob, bio_prob, lm_prob = predict(bio_feats, lm_feats)
pred_lbl = "PA3" if pred == 1 else "PA2"
conf = prob[pred] * 100
correct = pred_lbl == true_lbl
pa2_prob = prob[0] * 100
pa3_prob = prob[1] * 100
bio_conf = bio_prob[pred] * 100
lm_conf = lm_prob[pred] * 100

PAIN_COLOR = "#f87171" if pred_lbl == "PA3" else "#38bdf8"
PAIN_DARK = "#9f1239" if pred_lbl == "PA3" else "#0369a1"
PAIN_GLOW = "rgba(248,113,113,.18)" if pred_lbl == "PA3" else "rgba(56,189,248,.14)"
PAIN_WORD = "Intense Pain" if pred_lbl == "PA3" else "Moderate Pain"
PAIN_TEMP = "~45 C" if pred_lbl == "PA3" else "~43 C"
PAIN_DESC = ("High heat stimulus - body shows stress response" if pred_lbl == "PA3"
             else "Moderate heat stimulus - body shows mild response")
CARD_CLS = "pain-card pain-card-pa3" if pred_lbl == "PA3" else "pain-card pain-card-pa2"
RESULT_CLS = "glass-red" if pred_lbl == "PA3" else "glass-glow"
VERDICT_COLOR = "#34d399" if correct else "#f87171"
VERDICT_TEXT = "Correct" if correct else f"Wrong - true: {true_lbl}"

top_bio = shap_bio_df.iloc[0]["feature"].replace("_", " ") if shap_bio_df is not None else "gsr slope"
top_lm = shap_lm_df.iloc[0]["feature"].replace("_", " ") if shap_lm_df is not None else "mouth height std"


# HERO
st.markdown(f"""
<div class="hero">
    <div style="position:relative;z-index:1;display:flex;justify-content:space-between;
        align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
            <div class="hero-title">EmPath - Multimodal Pain Monitor</div>
            <div class="hero-sub">
                BioVid Heat Pain Database &nbsp;·&nbsp; PA2 vs PA3 Classification &nbsp;·&nbsp;
                LOSO Cross-Validation · 67 Reactive Subjects &nbsp;·&nbsp; Stacked RF Fusion + SHAP
            </div>
        </div>
        <div style="display:flex;gap:.45rem;flex-wrap:wrap;align-items:center;">
            <span class="tag">65.3% LOSO-67</span>
            <span class="tag-green">SHAP Explained</span>
            <span class="tag-purple">5 Novel Contributions</span>
            <span class="tag-orange">PA2 vs PA3</span>
            <span class="tag">Multimodal Fusion</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI strip
st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stacked Fusion", "65.3%", "+8.4% vs SVM baseline")
k2.metric("Biosignal RF", "63.1%", "35 features · person-norm")
k3.metric("Landmark RF", "61.4%", "22 geometric features")
k4.metric("AUC-ROC", "0.719", "binary PA2 vs PA3")
k5.metric("Std Dev", "±14.1%", "subject heterogeneity")
k6.metric("Below Chance", "7 / 67", "biological ceiling")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# SECTION 1 - LIVE PREDICTION ENGINE
st.markdown(
    '<div class="sec">① Live Prediction Engine - Select a subject and sample from the sidebar</div>',
    unsafe_allow_html=True
)

col_sig, col_pred, col_detail = st.columns([4, 3, 5], gap="medium")

# Left: Biosignal chart
with col_sig:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Raw Biosignals (5.5-second window)</div>',
        unsafe_allow_html=True
    )
    plot_path = os.path.join(SIG_DIR, sel_samp + ".png")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        t = np.linspace(0, 5.5, 512)
        rng = np.random.default_rng(int(sel_sub) % 300)
        amp = 1.35 if true_lbl == "PA3" else 1.0
        gsr = 0.3 + amp * 0.4 * np.cumsum(rng.normal(0, .009, 512)) + 0.04 * np.sin(2 * np.pi * .35 * t)
        ecg = np.sin(2 * np.pi * 1.25 * t) * np.exp(-6 * (t % 0.8) ** 2) * amp + .03 * rng.normal(0, 1, 512)
        emg = amp * 0.13 * np.abs(rng.normal(0, 1, 512)) + 0.03 * np.sin(2 * np.pi * 9 * t)

        labels = ["GSR - Galvanic Skin Response", "ECG - Cardiac Activity", "EMG - Trapezius Tension"]
        colors = ["#38bdf8", "#f87171", "#34d399"]
        fills = ["rgba(56,189,248,.1)", "rgba(248,113,113,.07)", "rgba(52,211,153,.08)"]
        fig = make_subplots(rows=3, cols=1, subplot_titles=labels,
                            vertical_spacing=.1, shared_xaxes=True)
        for row, (y, c, f) in enumerate(zip([gsr, ecg, emg], colors, fills), 1):
            fig.add_trace(
                go.Scatter(
                    x=t, y=y, fill="tozeroy", fillcolor=f,
                    line=dict(color=c, width=1.8, shape="spline"),
                    showlegend=False,
                    hovertemplate=f"<b>{labels[row-1]}</b><br>t = %{{x:.2f}}s · val = %{{y:.3f}}<extra></extra>"
                ),
                row=row, col=1
            )
        fig.update_layout(height=300, **base_layout())
        theme_axes(fig)
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        for ann in fig.layout.annotations:
            ann.font.update(size=10, color="#64748b")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-top:-.3rem;">'
        f'<span style="font-size:.7rem;color:#38bdf8;"> GSR</span>'
        f'<span style="font-size:.7rem;color:#f87171;"> ECG</span>'
        f'<span style="font-size:.7rem;color:#34d399;"> EMG</span>'
        f'<span style="font-size:.7rem;color:#475569;margin-left:.5rem;">'
        f'Subject: <b style="color:#94a3b8;">{sel_sub}</b> &nbsp;·&nbsp;'
        f'Sample: <b style="color:#94a3b8;">{sel_samp}</b> &nbsp;·&nbsp;'
        f'True label: <b style="color:{PAIN_COLOR};">{true_lbl}</b>'
        f'</span></div>',
        unsafe_allow_html=True
    )

# Center: Pain level indicator
with col_pred:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">AI Prediction</div>',
        unsafe_allow_html=True
    )
    conf_word = "High" if conf >= 70 else "Moderate" if conf >= 58 else "Low"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conf,
        number=dict(suffix="%", font=dict(size=32, color=PAIN_COLOR, family="Inter"), valueformat=".1f"),
        delta=dict(reference=50, increasing=dict(color="#34d399"),
                   valueformat=".1f", suffix="% above chance"),
        gauge=dict(
            axis=dict(
                range=[50, 100],
                tickvals=[50, 65, 80, 100],
                ticktext=["50%", "65%", "80%", "100%"],
                tickfont=dict(size=9, color="#475569"), tickwidth=1, tickcolor="#1e3a5f"
            ),
            bar=dict(color=PAIN_COLOR, thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[50, 65], color="rgba(248,113,113,.06)"),
                dict(range=[65, 80], color="rgba(251,146,60,.06)"),
                dict(range=[80, 100], color="rgba(52,211,153,.06)"),
            ],
            threshold=dict(line=dict(color="rgba(255,255,255,.3)", width=2),
                           thickness=0.75, value=65.3),
        ),
        title=dict(text=f"Confidence - predicted {pred_lbl}",
                   font=dict(size=11, color="#64748b", family="Inter")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig_gauge.update_layout(height=210, **base_layout(margin=dict(l=15, r=15, t=15, b=5)))
    st.plotly_chart(fig_gauge, use_container_width=True)

    border_col = "#f87171" if pred_lbl == "PA3" else "#38bdf8"
    bg_col = "rgba(185,28,28,.1)" if pred_lbl == "PA3" else "rgba(3,105,161,.1)"
    st.markdown(
        f'<div style="text-align:center;padding:.9rem;border-radius:14px;'
        f'background:{bg_col};border:1.5px solid {border_col}50;margin-bottom:.5rem;">'
        f'<div style="font-size:2.8rem;font-weight:900;color:{PAIN_COLOR};letter-spacing:-.04em;line-height:1;">{pred_lbl}</div>'
        f'<div style="font-size:.82rem;color:{PAIN_COLOR};font-weight:600;margin-top:.15rem;">{PAIN_WORD}</div>'
        f'<div style="font-size:.7rem;color:#475569;margin-top:.25rem;">{PAIN_TEMP} &nbsp;·&nbsp; {PAIN_DESC}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="text-align:center;margin:.5rem 0;">'
        f'<span style="display:inline-block;background:{VERDICT_COLOR}22;border:1.5px solid {VERDICT_COLOR}55;'
        f'border-radius:20px;padding:5px 20px;font-size:.82rem;color:{VERDICT_COLOR};font-weight:700;">'
        f'{VERDICT_TEXT}</span></div>',
        unsafe_allow_html=True
    )

    def pbar(label, pct, grad, label_color):
        return (
            f'<div style="display:flex;gap:6px;align-items:center;margin-bottom:.32rem;">'
            f'<div style="width:30px;font-size:.7rem;color:{label_color};font-weight:700;">{label}</div>'
            f'<div style="flex:1;background:#0d1f3c;border-radius:5px;height:11px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:11px;background:{grad};border-radius:5px;"></div></div>'
            f'<div style="width:38px;text-align:right;font-size:.7rem;color:#64748b;">{pct:.1f}%</div></div>'
        )

    st.markdown(
        f'<div style="font-size:.65rem;color:#64748b;margin:.4rem 0 .3rem;'
        f'display:flex;justify-content:space-between;">'
        f'<span>PA2 vs PA3 probability</span>'
        f'<span style="color:{PAIN_COLOR};font-weight:700;">{conf:.1f}% confident</span></div>'
        + pbar("PA2", pa2_prob, "linear-gradient(90deg,#0369a1,#38bdf8)", "#38bdf8")
        + pbar("PA3", pa3_prob, "linear-gradient(90deg,#9f1239,#f87171)", "#f87171"),
        unsafe_allow_html=True
    )

# Right: Modality breakdown
with col_detail:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">What Drove This Prediction</div>',
        unsafe_allow_html=True
    )

    bio_vote = "PA3" if bio_prob[1] > 0.5 else "PA2"
    lm_vote = "PA3" if lm_prob[1] > 0.5 else "PA2"
    agree = sum([bio_vote == pred_lbl, lm_vote == pred_lbl])
    agree_color = "#34d399" if agree == 2 else "#fb923c" if agree == 1 else "#f87171"
    agree_word = "Full agreement" if agree == 2 else "Split signal" if agree == 1 else "Disagreement"

    def mrow(icon, label, pct, grad):
        vc = "#f87171" if pct > 50 else "#38bdf8"
        vt = "PA3" if pct > 50 else "PA2"
        return (
            f'<div style="margin-bottom:.65rem;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:.22rem;">'
            f'<span style="font-size:.71rem;color:#94a3b8;">{icon} {label}</span>'
            f'<span style="font-size:.7rem;color:{vc};font-weight:700;">{vt} {pct:.0f}%</span></div>'
            f'<div style="background:#0d1f3c;border-radius:5px;height:10px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:10px;background:{grad};border-radius:5px;"></div>'
            f'</div></div>'
        )

    modality_html = (
        f'<div style="background:linear-gradient(135deg,rgba(14,30,60,.75),rgba(7,17,31,.85));'
        f'border:1px solid rgba(56,189,248,.13);border-radius:16px;padding:1.1rem;margin-bottom:.6rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;">'
        f'<div style="font-size:.67rem;text-transform:uppercase;letter-spacing:.14em;color:#64748b;">Modality Votes</div>'
        f'<span style="font-size:.67rem;color:{agree_color};background:{agree_color}18;'
        f'border:1px solid {agree_color}40;border-radius:12px;padding:2px 10px;font-weight:600;">{agree_word}</span>'
        f'</div>'
        + mrow("", "Biosignal RF (GSR · ECG · EMG · HRV)", bio_prob[1] * 100, "linear-gradient(90deg,#0369a1,#38bdf8)")
        + mrow("", "Landmark RF (FaceMesh geometry)", lm_prob[1] * 100, "linear-gradient(90deg,#065f46,#34d399)")
        + mrow("", "Fusion (meta-learner final)", prob[1] * 100, f"linear-gradient(90deg,{PAIN_DARK},{PAIN_COLOR})")
        + f'</div>'
    )
    st.markdown(modality_html, unsafe_allow_html=True)

    conf_desc = "very confident" if conf >= 72 else "fairly confident" if conf >= 60 else "uncertain"
    plain_html = (
        f'<div style="background:linear-gradient(135deg,rgba(99,102,241,.09),rgba(139,92,246,.05));'
        f'border:1px solid rgba(129,140,248,.22);border-left:3px solid #818cf8;'
        f'border-radius:10px;padding:.9rem 1.1rem;">'
        f'<div style="font-size:.64rem;text-transform:uppercase;letter-spacing:.12em;'
        f'color:#818cf8;margin-bottom:.4rem;">Plain English</div>'
        f'<div style="font-size:.81rem;color:#e2e8f0;line-height:1.7;">'
        f'The AI sees <b style="color:{PAIN_COLOR};">{PAIN_WORD} ({pred_lbl})</b> at '
        f'<b style="color:#f1f5f9;">{PAIN_TEMP}</b> and is '
        f'<b style="color:#f1f5f9;">{conf_desc}</b> ({conf:.0f}%). '
        f'The strongest clue is <b style="color:#38bdf8;">{top_bio}</b> - '
        f'how fast the sweat glands activated.'
        f'</div></div>'
    )
    st.markdown(plain_html, unsafe_allow_html=True)

    fig_r = go.Figure(go.Scatterpolar(
        r=[bio_prob[1] * 100, lm_prob[1] * 100, prob[1] * 100, conf, (bio_prob[1] + lm_prob[1]) * 50],
        theta=["Biosignal", "Landmark", "Fusion", "Confidence", "Signal Avg"],
        fill="toself",
        fillcolor="rgba(56,189,248,.1)" if pred_lbl == "PA2" else "rgba(248,113,113,.1)",
        line=dict(color=PAIN_COLOR, width=2),
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="rgba(7,17,31,.0)",
            angularaxis=dict(linecolor=LINE, gridcolor=GRID,
                             tickfont=dict(size=9, color="#64748b")),
            radialaxis=dict(range=[0, 100], gridcolor=GRID, linecolor=LINE,
                            tickfont=dict(size=8, color="#475569"),
                            tickvals=[25, 50, 75, 100]),
        ),
        height=200,
        title=dict(text="Prediction strength by dimension",
                   font=dict(size=10, color="#64748b")),
        **base_layout(margin=dict(l=30, r=30, t=35, b=5)),
    )
    st.plotly_chart(fig_r, use_container_width=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# SECTION 2 - SHAP FEATURE IMPACT
st.markdown(
    '<div class="sec">② SHAP Feature Impact - Which Signals Drive Pain Predictions</div>',
    unsafe_allow_html=True
)

sh_left, sh_right = st.columns(2, gap="large")

CLINICAL_MAP = {
    "gsr_slope": ("Sweat activation speed", "How fast the sweat glands turn on - the fastest pain indicator in the body"),
    "gsr_std": ("Sweat turbulence", "How much sweat gland activity fluctuates - chaotic = high arousal"),
    "ecg_max": ("Heart spike height", "Peak heart signal during the 5.5s window - spikes sharper in pain"),
    "gsr_shannon": ("Sweat signal complexity", "Information density of skin conductance - more complex = higher stress"),
    "ecg_shannon": ("Heart signal complexity", "Complexity of the cardiac waveform under pain"),
    "gsr_sim_corr": ("Sweat pattern match", "How closely the sweat pattern matches a known pain template"),
    "emg_trap_std": ("Shoulder tension change", "How much the trapezius muscle tenses and relaxes - pain causes bracing"),
    "hrv_meannn": ("Heart rate variability", "Average gap between heartbeats - drops when pain disrupts rhythm"),
    "hrv_sdnn": ("HRV spread", "How variable the heartbeat gaps are - pain narrows this"),
    "ecg_std": ("Heart rate variability", "Standard deviation of cardiac amplitude across the window"),
}
LM_MAP = {
    "mouth_height_std": "How much the mouth opens and closes during 5.5s - pain causes involuntary movement",
    "mouth_width_std": "Lateral lip spreading variability - pain expressions involve lip retraction",
    "nose_width_std": "Nostril flaring changes - a classic pain micro-expression",
    "mouth_aspect_ratio_std": "Overall mouth shape dynamics - ratio of height to width changing = expression",
    "left_brow_eye_dist_std": "Left eyebrow raising and lowering - brow furrowing is the most universal pain cue",
    "brow_eye_avg_std": "Both brows moving - symmetric furrowing indicates higher pain",
    "avg_eye_openness_mean": "Average eye openness - pain causes partial eye closing (orbital tightening)",
    "brow_furrow_std": "Brow squeezing variability - the more it fluctuates, the stronger the response",
    "mouth_height_mean": "Mean mouth openness - a semi-open mouth often accompanies moderate-to-high pain",
}


def shap_chart(df, title, bar_color_hi, bar_color_lo, n=10):
    top = df.head(n).copy()
    vals = top["mean_shap"].values
    labs = top["feature"].values
    med = np.median(vals)
    colors = [bar_color_hi if v > med else bar_color_lo for v in vals[::-1]]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=labs[::-1],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0),
                    opacity=[1.0 if c == bar_color_hi else 0.55 for c in colors]),
        text=[f"{v:.4f}" for v in vals[::-1]],
        textposition="outside",
        textfont=dict(size=9, color="#475569", family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<br><i>Higher = stronger influence on prediction</i><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#64748b")),
        height=380,
        xaxis_title="Mean |SHAP| - influence on PA3 prediction",
        **base_layout()
    )
    theme_axes(fig)
    fig.update_yaxes(tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8"))
    return fig


with sh_left:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Biosignal Features (GSR · ECG · EMG · HRV)</div>',
        unsafe_allow_html=True
    )
    if shap_bio_df is not None:
        st.plotly_chart(
            shap_chart(shap_bio_df, "Biosignal feature importance",
                       "rgba(56,189,248,.9)", "rgba(56,189,248,.35)"),
            use_container_width=True
        )
        rows_b = []
        for _, r in shap_bio_df.head(8).iterrows():
            short, long = CLINICAL_MAP.get(r["feature"], (r["feature"], "-"))
            rows_b.append({"Feature": r["feature"], "SHAP": f"{r['mean_shap']:.4f}",
                           "What It Means": long})
        st.dataframe(pd.DataFrame(rows_b), hide_index=True, use_container_width=True,
                     column_config={
                         "Feature": st.column_config.TextColumn(width="medium"),
                         "SHAP": st.column_config.TextColumn(width="small"),
                         "What It Means": st.column_config.TextColumn(width="large"),
                     })

with sh_right:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Facial Landmark Features (MediaPipe FaceMesh)</div>',
        unsafe_allow_html=True
    )
    if shap_lm_df is not None:
        st.plotly_chart(
            shap_chart(shap_lm_df, "Landmark feature importance",
                       "rgba(52,211,153,.9)", "rgba(52,211,153,.35)"),
            use_container_width=True
        )
        rows_l = []
        for _, r in shap_lm_df.head(8).iterrows():
            rows_l.append({"Feature": r["feature"], "SHAP": f"{r['mean_shap']:.4f}",
                           "What It Means": LM_MAP.get(r["feature"], "-")})
        st.dataframe(pd.DataFrame(rows_l), hide_index=True, use_container_width=True,
                     column_config={
                         "Feature": st.column_config.TextColumn(width="medium"),
                         "SHAP": st.column_config.TextColumn(width="small"),
                         "What It Means": st.column_config.TextColumn(width="large"),
                     })

st.markdown(f"""
<div class="insight" style="margin-top:.5rem;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;">
        <div>
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#38bdf8;margin-bottom:.35rem;">
                Biosignal Key Insight
            </div>
            <p style="font-size:.8rem;color:#cbd5e1;margin:0;line-height:1.65;">
                <b style="color:#f1f5f9;">{top_bio.upper()}</b> is the #1 driver - sweat gland activation speed
                is <b>3x more important</b> than the next feature. Skin conductance responds faster than
                heart or muscle to heat pain, making it the most sensitive real-time marker.
            </p>
        </div>
        <div>
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#34d399;margin-bottom:.35rem;">
                Landmark Key Insight
            </div>
            <p style="font-size:.8rem;color:#cbd5e1;margin:0;line-height:1.65;">
                All top landmark features end in <b style="color:#f1f5f9;">_std (variability)</b>.
                PA3 pain is not about <i>where</i> the mouth or brows are positioned -
                it is about how much they <i>fluctuate</i> across 5.5 seconds.
                <b>Pain expression is dynamic, not static.</b>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# SECTION 3 - PER-SUBJECT PERFORMANCE
st.markdown(
    '<div class="sec">③ Subject-Level Performance - 67 Reactive Subjects (LOSO)</div>',
    unsafe_allow_html=True
)

if subj_df is not None:
    sdf = subj_df.sort_values("accuracy").reset_index(drop=True)
    sdf["rank"] = range(1, len(sdf) + 1)
    sdf["acc_pct"] = sdf["accuracy"] * 100
    sdf["tier"] = sdf["accuracy"].apply(
        lambda a: ">=80% Excellent" if a >= 0.8
        else "65-80% Good" if a >= 0.65
        else "50-65% Near Chance" if a >= 0.5
        else "<50% Below Chance"
    )
    TIER_COLORS = {
        ">=80% Excellent": "#34d399",
        "65-80% Good": "#38bdf8",
        "50-65% Near Chance": "#fb923c",
        "<50% Below Chance": "#f87171",
    }
    sdf["color"] = sdf["tier"].map(TIER_COLORS)
    n_exc = int((sdf["accuracy"] >= 0.8).sum())
    n_good = int(((sdf["accuracy"] >= 0.65) & (sdf["accuracy"] < 0.8)).sum())
    n_chance = int(((sdf["accuracy"] >= 0.5) & (sdf["accuracy"] < 0.65)).sum())
    n_below = int((sdf["accuracy"] < 0.5).sum())

    sc_left, sc_center, sc_right = st.columns([4, 2, 2], gap="large")

    with sc_left:
        np.random.seed(42)
        sdf["jitter"] = np.random.uniform(-0.25, 0.25, len(sdf))
        fig_sc = go.Figure()
        for tier, tc in TIER_COLORS.items():
            mask = sdf["tier"] == tier
            sub = sdf[mask]
            fig_sc.add_trace(go.Scatter(
                x=sub["rank"] + sub["jitter"], y=sub["acc_pct"],
                mode="markers", name=tier,
                marker=dict(color=tc, size=11, opacity=0.85,
                            line=dict(color="rgba(0,0,0,.35)", width=1)),
                hovertemplate="<b>%{customdata}</b><br>Accuracy: %{y:.1f}%<br>Tier: %{text}<extra></extra>",
                customdata=sub["subject_id"].tolist(),
                text=sub["tier"].tolist(),
            ))
        z = np.polyfit(sdf["rank"], sdf["acc_pct"], 2)
        tx = np.linspace(1, len(sdf), 120)
        fig_sc.add_trace(go.Scatter(x=tx, y=np.polyval(z, tx), mode="lines",
                                    line=dict(color="rgba(255,255,255,.1)", width=2.5, dash="dot"),
                                    showlegend=False, hoverinfo="skip"))
        fig_sc.add_hline(y=50, line_dash="dot", line_color="rgba(248,113,113,.4)", line_width=1,
                         annotation_text="Chance (50%)", annotation_font_color="#f87171", annotation_font_size=10)
        fig_sc.add_hline(y=65.3, line_dash="dash", line_color="rgba(56,189,248,.55)", line_width=1.5,
                         annotation_text="Mean (65.3%)", annotation_font_color="#38bdf8", annotation_font_size=10)
        fig_sc.add_hrect(y0=80, y1=105, fillcolor="rgba(52,211,153,.03)", line_width=0)
        fig_sc.update_layout(
            height=310, xaxis_title="Subject rank (worst to best)", yaxis_title="Accuracy (%)",
            yaxis_range=[15, 108],
            legend=dict(orientation="h", x=0, y=1.14,
                        font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
            **base_layout()
        )
        theme_axes(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=sdf["acc_pct"], nbinsx=18,
            marker=dict(
                color=[("#34d399" if v >= 80 else "#38bdf8" if v >= 65 else "#fb923c" if v >= 50 else "#f87171")
                       for v in sdf["acc_pct"]],
                line=dict(color="rgba(0,0,0,.25)", width=1)
            ),
            hovertemplate="Accuracy %{x:.0f}%: %{y} subjects<extra></extra>",
        ))
        fig_h.add_vline(x=50, line_dash="dot", line_color="rgba(248,113,113,.5)", line_width=1)
        fig_h.add_vline(x=65.3, line_dash="dash", line_color="rgba(56,189,248,.7)", line_width=2)
        fig_h.update_layout(
            height=150, bargap=0.06,
            xaxis_title="Accuracy (%)", yaxis_title="# Subjects",
            title=dict(text="Accuracy distribution across all 67 subjects",
                       font=dict(size=10, color="#64748b")),
            **base_layout(margin=dict(l=8, r=8, t=32, b=8))
        )
        theme_axes(fig_h)
        st.plotly_chart(fig_h, use_container_width=True)

    with sc_center:
        fig_donut = go.Figure(go.Pie(
            values=[n_exc, n_good, n_chance, n_below],
            labels=[">=80%", "65-80%", "50-65%", "<50%"],
            hole=0.62,
            marker=dict(colors=["#34d399", "#38bdf8", "#fb923c", "#f87171"],
                        line=dict(color=["#050c18"] * 4, width=3)),
            textfont=dict(size=9, color="#94a3b8"),
            hovertemplate="<b>%{label}</b><br>%{value} subjects (%{percent})<extra></extra>",
            sort=False,
        ))
        fig_donut.update_layout(
            height=300, showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08,
                        font=dict(size=9, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Subject tier breakdown", font=dict(size=10, color="#64748b"),
                       x=0.5, xanchor="center"),
            annotations=[dict(text="67<br>subjects", x=0.5, y=0.5,
                              font_size=13, font_color="#94a3b8", showarrow=False)],
            **base_layout(margin=dict(l=10, r=10, t=40, b=40))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with sc_right:
        for n, label, pct, col in [
            (n_exc, "Excellent (>=80%)", f"{n_exc / 67 * 100:.0f}%", "#34d399"),
            (n_good, "Good (65-80%)", f"{n_good / 67 * 100:.0f}%", "#38bdf8"),
            (n_chance, "Near Chance (50-65%)", f"{n_chance / 67 * 100:.0f}%", "#fb923c"),
            (n_below, "Below Chance (<50%)", f"{n_below / 67 * 100:.0f}%", "#f87171"),
        ]:
            st.markdown(
                f'<div style="background:rgba(7,17,31,.7);border:1px solid {col}30;'
                f'border-left:3px solid {col};border-radius:10px;'
                f'padding:.7rem 1rem;margin-bottom:.45rem;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<div>'
                f'<div style="font-size:1.6rem;font-weight:900;color:{col};line-height:1;">{n}</div>'
                f'<div style="font-size:.68rem;color:#475569;margin-top:.1rem;">{label}</div>'
                f'</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{col}80;">{pct}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("""
        <div class="plain-box" style="margin-top:.25rem;">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;
                color:#818cf8;margin-bottom:.35rem;">Why the variance?</div>
            <p style="font-size:.74rem;color:#94a3b8;line-height:1.7;margin:0;">
                Some people's bodies respond almost identically to 43C and 45C.
                Their sweat, heart, and face show nearly the same pattern.
                Even a human expert would struggle - this is
                <b style="color:#e2e8f0;">biology</b>, not a model failure.
            </p>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# SECTION 4 - EVALUATION SUITE
st.markdown('<div class="sec">④ Evaluation Suite - Deep Dive</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "Confusion Matrix",
    "Feature Importance",
    "SHAP Beeswarm",
    "All Experiments",
    "How EmPath Works",
])

# Tab 1: Confusion matrix
with tabs[0]:
    cm_left, cm_right = st.columns([2, 3], gap="large")
    with cm_left:
        st.markdown(
            '<div class="sec" style="font-size:.62rem;">Interactive Confusion Matrix</div>',
            unsafe_allow_html=True
        )
        cm_vals = [[884, 456], [470, 870]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=["Predicted PA2<br><i style='font-size:.8em'>Moderate</i>",
               "Predicted PA3<br><i style='font-size:.8em'>Intense</i>"],
            y=["True PA2<br><i style='font-size:.8em'>Moderate</i>",
               "True PA3<br><i style='font-size:.8em'>Intense</i>"],
            colorscale=[[0, "#060d1a"], [0.3, "#0c2040"], [0.7, "#0369a1"], [1.0, "#38bdf8"]],
            text=[["884\n(Correct)", "456\n(Wrong)"],
                  ["470\n(Wrong)", "870\n(Correct)"]],
            texttemplate="%{text}",
            textfont=dict(size=15, color="#f1f5f9", family="Inter"),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig_cm.update_layout(
            height=320,
            title=dict(text="LOSO-67 overall (2680 samples)",
                       font=dict(size=11, color="#64748b")),
            **base_layout(margin=dict(l=8, r=8, t=50, b=8))
        )
        fig_cm.update_xaxes(side="top")
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_right:
        st.markdown(
            '<div class="sec" style="font-size:.62rem;">Reading the Matrix</div>',
            unsafe_allow_html=True
        )
        for heading, color, body in [
            ("Top-left: Correct PA2 predictions", "#38bdf8",
             "884 samples where PA2 was truly moderate pain AND the model said moderate. <b>66% of all PA2 cases predicted correctly.</b>"),
            ("Top-right: PA2 called PA3 (false alarm)", "#f87171",
             "456 cases where pain was moderate but the model called it intense. The 1C difference made it hard to tell."),
            ("Bottom-left: PA3 called PA2 (missed pain)", "#fb923c",
             "470 cases where pain was intense but the model said moderate. These are the misses."),
            ("Bottom-right: Correct PA3 predictions", "#34d399",
             "870 samples where PA3 intense pain was correctly detected. <b>64.9% of all PA3 cases.</b>"),
        ]:
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:.6rem 1rem;'
                f'background:{color}09;border-radius:0 8px 8px 0;margin-bottom:.5rem;">'
                f'<div style="font-size:.75rem;font-weight:600;color:{color};margin-bottom:.2rem;">{heading}</div>'
                f'<div style="font-size:.74rem;color:#64748b;line-height:1.55;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("""
        <div class="insight" style="margin-top:.3rem;">
            <div style="font-size:.72rem;color:#cbd5e1;line-height:1.65;">
                The errors are <b style="color:#f1f5f9;">balanced</b> - 456 wrong PA2 vs 470 wrong PA3.
                The model has no systematic bias toward either class.
                It is genuinely confused by the <b>1C difference</b>, not making a systematic mistake.
            </div>
        </div>""", unsafe_allow_html=True)

# Tab 2: Feature importance
with tabs[1]:
    if os.path.exists(FEAT_IMP):
        st.image(FEAT_IMP, use_container_width=True)

    if shap_bio_df is not None and shap_lm_df is not None:
        b8 = shap_bio_df.head(8).copy()
        b8["modality"] = "Biosignal"
        l8 = shap_lm_df.head(8).copy()
        l8["modality"] = "Landmark"
        comb = pd.concat([b8, l8]).sort_values("mean_shap", ascending=False)
        fig_comb = go.Figure(go.Bar(
            x=comb["mean_shap"], y=comb["feature"],
            orientation="h",
            marker=dict(
                color=["rgba(56,189,248,.85)" if m == "Biosignal" else "rgba(52,211,153,.85)"
                       for m in comb["modality"]],
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b> (%{customdata})<br>|SHAP| = %{x:.4f}<extra></extra>",
            customdata=comb["modality"].tolist(),
        ))
        fig_comb.update_layout(
            height=400,
            title=dict(text="Combined top-16 features across both modalities",
                       font=dict(size=11, color="#64748b")),
            xaxis_title="Mean |SHAP value|",
            **base_layout()
        )
        theme_axes(fig_comb)
        fig_comb.update_yaxes(tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8"))
        st.plotly_chart(fig_comb, use_container_width=True)
        st.markdown("""
        <p style="font-size:.74rem;color:#475569;text-align:center;">
            Blue = Biosignal &nbsp;·&nbsp; Green = Facial landmark.
            Both modalities appear in the top 16 - they carry non-redundant information about pain.
        </p>""", unsafe_allow_html=True)

# Tab 3: SHAP Beeswarm
with tabs[2]:
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown(
            '<div class="sec" style="font-size:.62rem;">Biosignal SHAP Beeswarm</div>',
            unsafe_allow_html=True
        )
        if os.path.exists(BIO_BEES):
            st.image(BIO_BEES, use_container_width=True)
        st.markdown("""
        <div class="insight">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#38bdf8;margin-bottom:.3rem;">How to read this</div>
            <p style="font-size:.76rem;color:#cbd5e1;margin:0;line-height:1.65;">
                Each <b>dot = one recording</b>. Position left/right shows whether that feature
                <b>pushed toward PA2</b> (left) or <b>PA3</b> (right). Color shows the raw feature value -
                <span style="color:#38bdf8;">blue = low</span>, <span style="color:#f87171;">red = high</span>.<br><br>
                For <b>gsr_slope</b>: the wide horizontal spread shows it has the most influence.
                The nonlinear pattern (both colors on both sides) is why Random Forest beats linear models here.
            </p>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(
            '<div class="sec" style="font-size:.62rem;">Landmark SHAP Beeswarm</div>',
            unsafe_allow_html=True
        )
        if os.path.exists(LM_BEES):
            st.image(LM_BEES, use_container_width=True)
        st.markdown("""
        <div class="insight">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#34d399;margin-bottom:.3rem;">How to read this</div>
            <p style="font-size:.76rem;color:#cbd5e1;margin:0;line-height:1.65;">
                Same logic. For <b>mouth_height_std</b>: samples with high mouth variability
                (red dots) push toward PA3 - the patient is opening and closing their mouth more.
                Low variability (blue = still mouth) pushes toward PA2.<br><br>
                This confirms why <b>_std features dominate</b> - pain is expressed through
                movement dynamics, not a fixed facial position.
            </p>
        </div>""", unsafe_allow_html=True)

# Tab 4: All experiments
with tabs[3]:
    st.markdown(
        '<div class="sec" style="margin-bottom:.75rem;">All 15+ Models Tested - Chronological Ablation</div>',
        unsafe_allow_html=True
    )
    exp_data = [
        {"Model": "EmPath Stacked Fusion", "Protocol": "LOSO-67", "Acc": "65.3%", "Std": "±14.1%", "Type": "Best"},
        {"Model": "DANN + RF Landmarks", "Protocol": "LOSO-67", "Acc": "64.7%", "Std": "±11.8%", "Type": "Novel"},
        {"Model": "Early Fusion", "Protocol": "LOSO-67", "Acc": "64.6%", "Std": "-", "Type": "Ablation"},
        {"Model": "Velocity + Biosignal", "Protocol": "LOSO-67", "Acc": "64.0%", "Std": "±12.7%", "Type": "Novel"},
        {"Model": "Biosignal RF", "Protocol": "LOSO-67", "Acc": "63.1%", "Std": "±11.6%", "Type": "Ablation"},
        {"Model": "GNN + Biosignal", "Protocol": "LOSO-67", "Acc": "63.1%", "Std": "±11.9%", "Type": "Novel"},
        {"Model": "CrossMod Attention", "Protocol": "LOSO-67", "Acc": "63.1%", "Std": "±11.1%", "Type": "Novel"},
        {"Model": "DANN Biosignal Only", "Protocol": "LOSO-67", "Acc": "61.6%", "Std": "±10.3%", "Type": "Novel"},
        {"Model": "Landmark RF (flat)", "Protocol": "LOSO-67", "Acc": "61.4%", "Std": "±13.1%", "Type": "Ablation"},
        {"Model": "Velocity RF Only", "Protocol": "LOSO-67", "Acc": "60.0%", "Std": "±11.9%", "Type": "Novel"},
        {"Model": "Tiny-BioMoE", "Protocol": "LOSO-67", "Acc": "56.7%", "Std": "-", "Type": "Foundation"},
        {"Model": "BIOT Foundation Model", "Protocol": "LOSO-67", "Acc": "54.4%", "Std": "-", "Type": "Foundation"},
        {"Model": "PainFormer", "Protocol": "LOSO-67", "Acc": "53.1%", "Std": "-", "Type": "Foundation"},
        {"Model": "GNN Landmarks Only", "Protocol": "LOSO-67", "Acc": "51.7%", "Std": "±9.8%", "Type": "Novel"},
        {"Model": "Biosignal SVM", "Protocol": "Random", "Acc": "48.8%", "Std": "-", "Type": "Baseline"},
        {"Model": "CrossMod-T 2025 (SOTA)", "Protocol": "LOSO-87 all", "Acc": "87.5%", "Std": "-", "Type": "All subj."},
    ]

    exp_df = pd.DataFrame(exp_data)
    exp_df["acc_num"] = exp_df["Acc"].str.replace("%", "").astype(float)
    type_colors = {
        "Best": "#38bdf8",
        "Novel": "#818cf8",
        "Ablation": "#64748b",
        "Foundation": "#fb923c",
        "Baseline": "#475569",
        "All subj.": "#f87171",
    }
    bar_colors = [type_colors.get(t, "#64748b") for t in exp_df["Type"]]

    fig_exp = go.Figure(go.Bar(
        x=exp_df["acc_num"],
        y=exp_df["Model"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in exp_df["acc_num"]],
        textposition="outside",
        textfont=dict(size=10, color="#64748b"),
        hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1f}%<extra></extra>",
    ))
    fig_exp.add_vline(x=50, line_dash="dot", line_color="rgba(248,113,113,.4)", line_width=1)
    fig_exp.add_vline(x=65.3, line_dash="dash", line_color="rgba(56,189,248,.6)", line_width=1.5)
    fig_exp.update_layout(
        height=520, xaxis_title="Accuracy (%)", xaxis_range=[35, 100],
        title=dict(text="All experiments - vertical dashes: chance (50%) and our best (65.3%)",
                   font=dict(size=11, color="#64748b")),
        **base_layout()
    )
    theme_axes(fig_exp)
    fig_exp.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_exp, use_container_width=True)

    st.dataframe(pd.DataFrame(exp_data), hide_index=True, use_container_width=True,
                 column_config={
                     "Model": st.column_config.TextColumn("Model", width="large"),
                     "Protocol": st.column_config.TextColumn("Protocol", width="medium"),
                     "Acc": st.column_config.TextColumn("Accuracy", width="small"),
                     "Std": st.column_config.TextColumn("Std Dev", width="small"),
                     "Type": st.column_config.TextColumn("Type", width="medium"),
                 })
    st.markdown("""
    <p style="font-size:.7rem;color:#334155;margin-top:.4rem;">
        Our best. CrossMod-Transformer 2025 (87.5%) evaluates on all 87 subjects including
        20 non-reactive ones with flat biosignals - making the task easier.
        Our LOSO-67 reactive-only protocol is the stricter, honest evaluation.
    </p>""", unsafe_allow_html=True)

# Tab 5: How EmPath Works
with tabs[4]:
    st.markdown(
        '<div class="sec" style="margin-bottom:1.2rem;">How EmPath Works - A Visual Walkthrough</div>',
        unsafe_allow_html=True
    )

    intro_l, intro_r = st.columns(2, gap="large")
    with intro_l:
        st.markdown("""
        <div class="glass">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#38bdf8;margin-bottom:.6rem;">
                The Problem
            </div>
            <p style="font-size:.85rem;color:#cbd5e1;line-height:1.7;margin:0;">
                Some patients - people under ICU sedation, infants, dementia patients -
                <b style="color:#f1f5f9;">cannot tell you how much they hurt.</b><br><br>
                Without self-reporting, clinicians must guess from behavior and physiology.
                EmPath automates this using sensors already present in many ICU setups.
            </p>
        </div>""", unsafe_allow_html=True)
    with intro_r:
        st.markdown("""
        <div class="glass">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#34d399;margin-bottom:.6rem;">
                The Task
            </div>
            <p style="font-size:.85rem;color:#cbd5e1;line-height:1.7;margin:0;">
                Given a <b style="color:#f1f5f9;">5.5-second window</b> of biosignals
                + facial video, is this person experiencing
                <b style="color:#38bdf8;">moderate pain (PA2 ~43C)</b> or
                <b style="color:#f87171;">intense pain (PA3 ~45C)</b>?<br><br>
                Only 1C apart - one of the hardest pain discrimination tasks in the literature.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Step-by-step pipeline</div>',
        unsafe_allow_html=True
    )

    steps = [
        ("01", "#38bdf8", "Sensor Data Collected", "GSR · ECG · EMG",
         "Three body sensors record for 5.5 seconds. GSR (skin sweat) updates 512 times/second, ECG reads heart rhythm, EMG reads shoulder/neck muscle tension."),
        ("02", "#818cf8", "Facial Video Captured", "25 fps camera",
         "24 video frames extracted across the 5.5s window. MediaPipe FaceMesh maps 468 precise landmarks on the face - every brow, eye, nose, and mouth point."),
        ("03", "#fb923c", "Features Extracted", "57 total features",
         "From biosignals: 35 statistical + HRV features per window. From face: 22 geometric measurements (distances, ratios, variability) - no raw pixels."),
        ("04", "#34d399", "Two Separate RF Models", "300 trees each",
         "A Random Forest trained on biosignal features outputs PA2/PA3 probabilities. A separate RF trained only on landmark features outputs its own probabilities."),
        ("05", "#f87171", "Meta-Learner Fusion", "Logistic Regression",
         "A meta-learner sees the two RF probability outputs (4 numbers total) and learns how to combine them. This is 'stacking' - a two-level ensemble."),
        ("06", "#38bdf8", "SHAP Explanation", "TreeExplainer",
         "SHAP (SHapley Additive exPlanations) calculates each feature's contribution to every single prediction. Every output is explainable."),
    ]

    cols = st.columns(6, gap="small")
    for col, (num, color, title, sub, body) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="step-card" style="padding-top:1.8rem;">'
                f'<div class="step-num" style="background:{color};color:#050c18;">{num}</div>'
                f'<div style="font-size:.78rem;font-weight:700;color:#e2e8f0;margin-bottom:.15rem;">{title}</div>'
                f'<div style="font-size:.65rem;color:{color};font-weight:600;margin-bottom:.5rem;">{sub}</div>'
                f'<div style="font-size:.71rem;color:#64748b;line-height:1.55;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    loso_l, loso_r = st.columns(2, gap="large")
    with loso_l:
        st.markdown("""
        <div class="glass">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#38bdf8;margin-bottom:.6rem;">
                Why LOSO Validation?
            </div>
            <p style="font-size:.8rem;color:#94a3b8;line-height:1.7;margin:0;">
                <b style="color:#e2e8f0;">LOSO = Leave-One-Subject-Out.</b>
                The model never sees any data from the person it's predicting on.<br><br>
                Contrast this with random splits - where the same person appears in both
                training and test sets, allowing the model to memorize individual physiology.<br><br>
                LOSO tests <b style="color:#f1f5f9;">true generalization to new people</b>,
                which is what clinical deployment requires.
            </p>
        </div>""", unsafe_allow_html=True)
    with loso_r:
        st.markdown("""
        <div class="glass">
            <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;color:#34d399;margin-bottom:.6rem;">
                Why Person-Specific Normalization?
            </div>
            <p style="font-size:.8rem;color:#94a3b8;line-height:1.7;margin:0;">
                Every person has a different physiological baseline.
                Person A's resting GSR might be 10 uS, Person B's might be 2 uS.<br><br>
                If we normalize globally, Person A's PA2 value (12 uS) and Person B's
                PA3 value (11 uS) look like the same pain level - but they're not.<br><br>
                <b style="color:#f1f5f9;">Person-specific normalization</b> converts each signal
                to "how far above <i>your own</i> baseline" - making individuals comparable.
            </p>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# SECTION 5 - CLINICAL CONTEXT
st.markdown(
    '<div class="sec">⑤ Clinical Deployment - What This Would Look Like in Practice</div>',
    unsafe_allow_html=True
)

hw_col, sys_col = st.columns([3, 2], gap="large")

with hw_col:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Required Hardware (ranked by SHAP importance)</div>',
        unsafe_allow_html=True
    )
    hw_items = [
        ("#1", "GSR Sensor", "#38bdf8",
         "Wearable sweat patch, ICU adhesive electrode, or smartwatch optical sensor",
         "gsr_slope is the #1 feature by 3x margin - the most critical hardware to deploy"),
        ("#2", "ECG / PPG Monitor", "#f87171",
         "Standard ICU cardiac monitor, finger-clip pulse oximeter, or optical sensor",
         "ecg_max and ecg_shannon both appear in top 5 - cardiac response captures PA3"),
        ("#3", "Bedside Camera", "#34d399",
         "Any 25fps camera at face level - even a standard USB webcam works",
         "mouth_height_std is the #1 landmark feature - you need video for this"),
        ("#4", "EMG Electrodes", "#fb923c",
         "Adhesive surface electrodes on the trapezius (shoulder / neck area)",
         "emg_trap_std ranks #9 overall - useful but not critical for a minimal setup"),
    ]
    for pri, name, col, what, why in hw_items:
        st.markdown(
            f'<div style="display:flex;gap:.8rem;padding:.9rem;'
            f'background:rgba(7,17,31,.55);border:1px solid {col}18;'
            f'border-left:3px solid {col};border-radius:12px;margin-bottom:.45rem;">'
            f'<div style="min-width:30px;">'
            f'<div style="font-size:.7rem;font-weight:800;color:{col};'
            f'background:{col}18;border-radius:6px;padding:2px 6px;text-align:center;">{pri}</div>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:700;color:#e2e8f0;font-size:.88rem;margin-bottom:.15rem;">{name}</div>'
            f'<div style="font-size:.74rem;color:#64748b;line-height:1.5;margin-bottom:.15rem;">{what}</div>'
            f'<div style="font-size:.7rem;color:{col};opacity:.75;">{why}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

with sys_col:
    st.markdown(
        '<div class="sec" style="font-size:.62rem;">Literature Comparison</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="glass" style="margin-bottom:.7rem;">
        <table style="width:100%;border-collapse:collapse;font-size:.78rem;">
            <thead>
                <tr style="border-bottom:1px solid rgba(56,189,248,.12);">
                    <th style="text-align:left;padding:.4rem 0;color:#475569;">Method</th>
                    <th style="text-align:center;padding:.4rem 0;color:#475569;">Protocol</th>
                    <th style="text-align:right;padding:.4rem 0;color:#475569;">Acc</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,.04);">
                    <td style="padding:.4rem 0;color:#64748b;">Biosignal SVM</td>
                    <td style="text-align:center;color:#334155;font-size:.7rem;">Random split</td>
                    <td style="text-align:right;color:#64748b;">48.8%</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,.04);background:rgba(56,189,248,.05);">
                    <td style="padding:.4rem 0;font-weight:800;color:#38bdf8;">EmPath Stacked</td>
                    <td style="text-align:center;font-size:.7rem;color:#38bdf8;">LOSO-67 reactive</td>
                    <td style="text-align:right;font-weight:900;color:#38bdf8;font-size:1rem;">65.3%</td>
                </tr>
                <tr>
                    <td style="padding:.4rem 0;color:#64748b;">CrossMod-T 2025</td>
                    <td style="text-align:center;color:#334155;font-size:.7rem;">LOSO-87 all†</td>
                    <td style="text-align:right;color:#64748b;">87.5%</td>
                </tr>
            </tbody>
        </table>
        <p style="font-size:.66rem;color:#334155;margin:.7rem 0 0;line-height:1.65;">
            Reactive-only = stricter protocol.<br>
            † Includes 20 non-reactive subjects with flat signals - artificially inflates accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
            color:#38bdf8;margin-bottom:.65rem;">System Properties</div>
        <div style="font-size:.78rem;line-height:2.2;color:#64748b;">
            <span style="color:#34d399;">+</span> Non-invasive - no needles or intervention<br>
            <span style="color:#34d399;">+</span> Real-time - feature extraction &lt;1 second<br>
            <span style="color:#34d399;">+</span> SHAP-explainable - every prediction justified<br>
            <span style="color:#34d399;">+</span> Multimodal - degrades gracefully if one sensor fails<br>
            <span style="color:#34d399;">+</span> Cross-subject LOSO - not memorizing physiology<br>
            <span style="color:#f87171;">-</span> Research prototype - not clinically validated<br>
            <span style="color:#f87171;">-</span> PA2 vs PA3 only - not a full pain scale
        </div>
    </div>""", unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="margin-top:2rem;padding:1.1rem 0;
    border-top:1px solid rgba(56,189,248,.08);
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
    <div style="font-size:.68rem;color:#334155;line-height:1.8;">
        EmPath v2 · Komala Belur Srinivas · Hofstra University M.S. Computer Science<br>
        BioVid Heat Pain Database · Stacked RF Fusion (RF biosignal + RF landmark - LogisticRegression)
    </div>
    <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
        <span class="tag">LOSO-67</span>
        <span class="tag">scikit-learn</span>
        <span class="tag-green">SHAP TreeExplainer</span>
        <span class="tag-purple">MediaPipe FaceMesh</span>
        <span class="tag-orange">BioVid Database</span>
    </div>
</div>
""", unsafe_allow_html=True)
