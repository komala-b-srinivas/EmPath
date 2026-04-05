import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="EmPath Pain Detection",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "Models", "empath_model.pkl")
DEMO_PATH     = os.path.join(BASE_DIR, "Models", "demo_samples.csv")
PER_SUBJ_PATH = os.path.join(BASE_DIR, "Results", "error_analysis", "per_subject_accuracy.csv")
CONFUSION_IMG = os.path.join(BASE_DIR, "Results", "error_analysis", "confusion_matrix_final.png")
FEATURE_IMG   = os.path.join(BASE_DIR, "Results", "error_analysis", "feature_importance.png")
ROC_IMG       = os.path.join(BASE_DIR, "Results", "error_analysis", "roc_curve.png")
SHAP_BIO_IMG  = os.path.join(BASE_DIR, "Results", "error_analysis", "shap_biosignal.png")
SHAP_LM_IMG   = os.path.join(BASE_DIR, "Results", "error_analysis", "shap_landmarks.png")
BIOSIG_DIR    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/biosignals_filtered"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_demo_samples():
    return pd.read_csv(DEMO_PATH)

@st.cache_data
def load_per_subject():
    if os.path.exists(PER_SUBJ_PATH):
        return pd.read_csv(PER_SUBJ_PATH)
    return None

model   = load_model()
samples = load_demo_samples()
subj_df = load_per_subject()

def predict(bio_feats, lm_feats):
    bio = np.array(bio_feats).reshape(1, -1)
    lm  = np.array(lm_feats).reshape(1, -1)
    bio = (bio - model["global_bio_mean"]) / model["global_bio_std"]
    lm  = (lm  - model["global_lm_mean"])  / model["global_lm_std"]
    bio_prob  = model["rf_bio"].predict_proba(bio)
    lm_prob   = model["rf_lm"].predict_proba(lm)
    meta_feat = np.hstack([bio_prob, lm_prob])
    pred      = model["meta"].predict(meta_feat)[0]
    prob      = model["meta"].predict_proba(meta_feat)[0]
    return pred, prob, bio_prob[0], lm_prob[0]

def get_feature_values(sample_name):
    bio_cols  = model["bio_cols"]
    lm_cols   = model["lm_cols"]
    row       = samples[samples["sample_name"] == sample_name].iloc[0]
    bio_feats = [row[f"bio_{i}"] for i in range(len(bio_cols))]
    lm_feats  = [row[f"lm_{i}"]  for i in range(len(lm_cols))]
    return bio_feats, lm_feats

# ── Title ──────────────────────────────────────────────────────────────
st.title("EmPath Multimodal Pain Intensity Detection")
st.markdown("**Research Demo** | BioVid Heat Pain Database | PA2 vs PA3 Classification")
st.markdown("---")

# ── Clinical Motivation ────────────────────────────────────────────────
with st.expander("Why Automated Pain Assessment Matters", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **ICU Patients**

        Sedated or unconscious patients cannot self-report pain.
        Nurses assess every 4 hours pain goes undetected between checks.
        """)
    with col_b:
        st.markdown("""
        **Dementia Patients**

        Cognitive decline prevents reliable pain communication.
        Undertreated pain accelerates functional decline.
        """)
    with col_c:
        st.markdown("""
        **Neonates**

        Infants cannot verbalize pain.
        Assessment relies on behavioral observation subjective and inconsistent.
        """)
    st.info("EmPath provides **continuous, objective, multimodal** pain monitoring "
            "using biosignals and facial analysis no patient cooperation required.")

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
st.sidebar.markdown("### Select Sample")

unique_subjects  = sorted(samples["subject_id"].unique())
selected_subject = st.sidebar.selectbox("Subject ID", unique_subjects)
subject_samples  = samples[
    samples["subject_id"] == selected_subject]["sample_name"].tolist()
selected_sample  = st.sidebar.selectbox("Sample", subject_samples)
true_label       = samples[
    samples["sample_name"] == selected_sample]["class_name"].values[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.metric("LOSO Accuracy", "65.3%")
st.sidebar.metric("AUC-ROC", "0.719")
st.sidebar.metric("F1 Score", "0.653")
st.sidebar.metric("Subjects", "67 reactive")
st.sidebar.text("Biosignals + Landmarks")
st.sidebar.markdown("---")
st.sidebar.warning(
    "**Generalization Notice**\n\n"
    "Results are from LOSO cross-validation on 67 BioVid subjects. "
    "**New subjects may perform differently** due to individual physiological differences."
)

# ── Get features before columns ────────────────────────────────────────
bio_feats, lm_feats = get_feature_values(selected_sample)
pred, prob, bio_prob, lm_prob = predict(bio_feats, lm_feats)
pred_label = "PA3" if pred == 1 else "PA2"
confidence = prob[pred] * 100
correct    = pred_label == true_label
color      = "#388e3c" if correct else "#d32f2f"
icon       = "Correct" if correct else "Incorrect"
bio_conf   = bio_prob[pred] * 100
lm_conf    = lm_prob[pred]  * 100

# ── Two column layout ──────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Biosignals")

    subject  = selected_sample.split("-")[0]
    bio_path = os.path.join(BIOSIG_DIR, subject, selected_sample + "_bio.csv")

    if os.path.exists(bio_path):
        raw_df = pd.read_csv(bio_path, sep="\t")
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("GSR (Skin Conductance)",
                            "ECG (Heart Rate)",
                            "EMG Corrugator (Facial Muscle)"),
            vertical_spacing=0.12
        )
        time = np.arange(len(raw_df)) / 512
        fig.add_trace(go.Scatter(x=time, y=raw_df["gsr"],
            line=dict(color="#1976d2", width=1.5), name="GSR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=raw_df["ecg"],
            line=dict(color="#d32f2f", width=1), name="ECG"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=raw_df["emg_corrugator"],
            line=dict(color="#388e3c", width=1), name="EMG"), row=3, col=1)
        fig.update_layout(height=400, showlegend=False,
                          margin=dict(l=0, r=0, t=40, b=0))
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Raw signal file not found.")
        sig_data    = {col: samples[samples["sample_name"] == selected_sample].iloc[0][f"bio_{i}"]
                       for i, col in enumerate(model["bio_cols"])}
        key_signals = ["gsr_mean", "gsr_std", "gsr_slope",
                       "ecg_mean", "ecg_std", "ecg_max",
                       "emg_corr_mean", "emg_corr_std"]
        key_vals    = [sig_data.get(k, 0) for k in key_signals]
        bar_colors  = ["#1976d2" if "gsr" in k else
                       "#d32f2f" if "ecg" in k else
                       "#388e3c" for k in key_signals]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=key_signals, y=key_vals, marker_color=bar_colors))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Prediction")

    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color};
                border-radius:10px; padding:20px; text-align:center;">
        <h1 style="color:{color}; margin:0;">{pred_label}</h1>
        <p style="color:{color}; margin:5px 0;">Predicted Pain Level</p>
        <h3 style="margin:10px 0;">{icon}  {confidence:.1f}% confidence</h3>
        <p>True label: <b>{true_label}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Modality Contributions")
    st.markdown("**Biosignals**")
    st.progress(int(bio_conf))
    st.caption(f"{bio_conf:.1f}% confidence for {pred_label}")
    st.markdown("**Facial Landmarks**")
    st.progress(int(lm_conf))
    st.caption(f"{lm_conf:.1f}% confidence for {pred_label}")

# ── Feature Importance ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Features Driving This Prediction")

bio_cols     = model["bio_cols"]
importances  = model["rf_bio"].feature_importances_
top_idx      = np.argsort(importances)[::-1][:8]
top_features = [bio_cols[i] for i in top_idx]
top_imp      = [importances[i] for i in top_idx]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=top_features, y=top_imp,
                      marker_color=["#1976d2"] * 8))
fig2.update_layout(height=300,
                   title="Top 8 Biosignal Feature Importances",
                   yaxis_title="Importance",
                   margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig2, use_container_width=True)

# ── Per-Subject Accuracy ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Per-Subject Model Performance")

if subj_df is not None:
    subj_sorted = subj_df.sort_values(
        "accuracy", ascending=False).reset_index(drop=True)

    colors_bar = []
    for acc in subj_sorted["accuracy"]:
        if acc >= 0.80:
            colors_bar.append("#388e3c")
        elif acc >= 0.65:
            colors_bar.append("#1976d2")
        elif acc >= 0.50:
            colors_bar.append("#f57c00")
        else:
            colors_bar.append("#d32f2f")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=list(range(len(subj_sorted))),
        y=subj_sorted["accuracy"] * 100,
        marker_color=colors_bar,
        hovertext=[f"Subject {int(s)}: {a*100:.1f}%"
                   for s, a in zip(subj_sorted["subject_id"],
                                   subj_sorted["accuracy"])],
        hoverinfo="text"
    ))
    fig3.add_hline(y=50,   line_dash="dash", line_color="red",
                   annotation_text="Chance (50%)")
    fig3.add_hline(y=65.3, line_dash="dash", line_color="blue",
                   annotation_text="Mean (65.3%)")
    fig3.update_layout(
        height=350,
        title="Per-Subject Accuracy sorted best to worst",
        xaxis_title="Subject rank",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Excellent (>80%)",
              f"{sum(subj_sorted['accuracy'] > 0.80)} subjects")
    c2.metric("Good (65-80%)",
              f"{sum((subj_sorted['accuracy'] >= 0.65) & (subj_sorted['accuracy'] <= 0.80))} subjects")
    c3.metric("Near chance",
              f"{sum((subj_sorted['accuracy'] >= 0.50) & (subj_sorted['accuracy'] < 0.65))} subjects")
    c4.metric("Below chance",
              f"{sum(subj_sorted['accuracy'] < 0.50)} subjects")

    st.info("""
    **Why does accuracy vary so much? (plus/minus 14.1% std)**

    - **Stoic subjects**: some people suppress physiological pain responses.
      Their GSR, ECG and facial muscles show minimal change even under moderate pain.

    - **Individual baseline differences**: despite person-specific normalization,
      some subjects have highly variable resting signals that mask pain responses.

    - **Genuine ambiguity**: PA2 and PA3 are adjacent pain levels separated by roughly 1 degree C.
      For some subjects this difference is physiologically undetectable.

    The 7 below-chance subjects represent genuine edge cases.
    Even clinical experts would struggle to distinguish their PA2 from PA3 responses.
    """)

# ── Evaluation Results ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Evaluation Results")

tab1, tab2, tab3, tab4 = st.tabs(
    ["ROC Curve", "Confusion Matrix", "SHAP Biosignals", "SHAP Landmarks"])

with tab1:
    if os.path.exists(ROC_IMG):
        st.image(ROC_IMG, caption="ROC Curve — AUC 0.719")
    else:
        st.info("ROC curve image not available")

with tab2:
    if os.path.exists(CONFUSION_IMG):
        st.image(CONFUSION_IMG, caption="Confusion Matrix — LOSO 67 Subjects")
    else:
        st.info("Confusion matrix image not available")

with tab3:
    if os.path.exists(SHAP_BIO_IMG):
        st.image(SHAP_BIO_IMG, caption="SHAP Feature Importance — Biosignals")
    else:
        st.info("SHAP biosignal image not available")

with tab4:
    if os.path.exists(SHAP_LM_IMG):
        st.image(SHAP_LM_IMG, caption="SHAP Feature Importance — Landmarks")
    else:
        st.info("SHAP landmark image not available")

# ── Model Performance Summary ──────────────────────────────────────────
st.markdown("---")
st.subheader("Model Performance Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Accuracy", "65.3%", "+8.4% vs baseline")
c2.metric("Biosignal Only", "63.1%", "+6.2% vs baseline")
c3.metric("Landmarks Only", "61.4%", "+4.5% vs baseline")
c4.metric("AUC-ROC", "0.719")

st.caption(
    "Subject-independent LOSO cross-validation on 67 reactive subjects. "
    "New subjects may perform differently than shown here.")

st.markdown("---")
st.caption("EmPath Capstone Project | BioVid Heat Pain Database | "
           "Stacked Fusion: RF (Bio) + RF (Landmarks) → Logistic Regression")