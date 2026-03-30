import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="'EmPath' Pain Detection",
    page_icon="🧠",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────────
MODEL_PATH  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Models/empath_model.pkl"
DEMO_PATH   = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Models/demo_samples.csv"
BIOSIG_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/biosignals_filtered"
LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67/landmarks_all67.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_demo_samples():
    return pd.read_csv(DEMO_PATH)

@st.cache_data
def load_landmarks():
    return pd.read_csv(LANDMARKS_CSV)

model   = load_model()
samples = load_demo_samples()
lm_df   = load_landmarks()

# ── Helper functions ───────────────────────────────────────────────────
def load_raw_signal(sample_name):
    parts       = sample_name.split("-")
    subject     = parts[0]
    bio_path    = os.path.join(BIOSIG_DIR, subject, sample_name + "_bio.csv")
    if os.path.exists(bio_path):
        return pd.read_csv(bio_path, sep="\t")
    return None

def predict(bio_feats, lm_feats, global_norm=True):
    bio = np.array(bio_feats).reshape(1, -1)
    lm  = np.array(lm_feats).reshape(1, -1)

    # Normalize
    bio = (bio - model["global_bio_mean"]) / model["global_bio_std"]
    lm  = (lm  - model["global_lm_mean"])  / model["global_lm_std"]

    bio_prob  = model["rf_bio"].predict_proba(bio)
    lm_prob   = model["rf_lm"].predict_proba(lm)
    meta_feat = np.hstack([bio_prob, lm_prob])
    pred      = model["meta"].predict(meta_feat)[0]
    prob      = model["meta"].predict_proba(meta_feat)[0]

    return pred, prob, bio_prob[0], lm_prob[0]

def get_feature_values(sample_name):
    bio_cols = model["bio_cols"]
    lm_cols  = model["lm_cols"]

    row = samples[samples["sample_name"] == sample_name].iloc[0]
    bio_feats = [row[f"bio_{i}"] for i in range(len(bio_cols))]
    lm_feats  = [row[f"lm_{i}"]  for i in range(len(lm_cols))]
    return bio_feats, lm_feats

# ── UI ─────────────────────────────────────────────────────────────────
st.title("🧠 EmPath Multimodal Pain Intensity Detection")
st.markdown("**Research Demo** | BioVid Heat Pain Database | PA2 vs PA3 Classification")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("### Select Sample")

unique_subjects = sorted(samples["subject_id"].unique())
selected_subject = st.sidebar.selectbox(
    "Subject ID", unique_subjects)

subject_samples = samples[
    samples["subject_id"] == selected_subject]["sample_name"].tolist()
selected_sample = st.sidebar.selectbox(
    "Sample", subject_samples)

true_label = samples[
    samples["sample_name"] == selected_sample]["class_name"].values[0]

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.metric("LOSO Accuracy", "65.3%")
st.sidebar.metric("Subjects", "67 reactive")
st.sidebar.text("Biosignals + Landmarks")

# ── Main content ───────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Biosignals")
    raw_df = load_raw_signal(selected_sample)

    if raw_df is not None:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("GSR (Skin Conductance)",
                            "ECG (Heart Rate)",
                            "EMG Corrugator (Facial Muscle)"),
            vertical_spacing=0.12
        )

        time = np.arange(len(raw_df)) / 512

        fig.add_trace(go.Scatter(
            x=time, y=raw_df["gsr"],
            line=dict(color="#1976d2", width=1.5),
            name="GSR"), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=time, y=raw_df["ecg"],
            line=dict(color="#d32f2f", width=1),
            name="ECG"), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=time, y=raw_df["emg_corrugator"],
            line=dict(color="#388e3c", width=1),
            name="EMG Corrugator"), row=3, col=1)

        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Raw signal file not found.")

with col2:
    st.subheader("Prediction")

    bio_feats, lm_feats = get_feature_values(selected_sample)
    pred, prob, bio_prob, lm_prob = predict(bio_feats, lm_feats)

    pred_label = "PA3" if pred == 1 else "PA2"
    confidence = prob[pred] * 100
    correct    = pred_label == true_label

    # Prediction box
    color = "#388e3c" if correct else "#d32f2f"
    icon  = "✅" if correct else "❌"

    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color};
                border-radius:10px; padding:20px; text-align:center;">
        <h1 style="color:{color}; margin:0;">{pred_label}</h1>
        <p style="color:{color}; margin:5px 0;">Predicted Pain Level</p>
        <h3 style="margin:10px 0;">{icon} {confidence:.1f}% confidence</h3>
        <p>True label: <b>{true_label}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Modality contributions
    st.markdown("#### Modality Contributions")
    bio_conf = bio_prob[pred] * 100
    lm_conf  = lm_prob[pred]  * 100

    st.markdown("**Biosignals**")
    st.progress(int(bio_conf))
    st.caption(f"{bio_conf:.1f}% confidence for {pred_label}")

    st.markdown("**Facial Landmarks**")
    st.progress(int(lm_conf))
    st.caption(f"{lm_conf:.1f}% confidence for {pred_label}")

# ── Feature importance bar ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Features Driving This Prediction")

bio_cols = model["bio_cols"]
lm_cols  = model["lm_cols"]

importances    = model["rf_bio"].feature_importances_
top_idx        = np.argsort(importances)[::-1][:8]
top_features   = [bio_cols[i] for i in top_idx]
top_importance = [importances[i] for i in top_idx]
top_values     = [bio_feats[i] for i in top_idx]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=top_features,
    y=top_importance,
    marker_color=["#1976d2"] * len(top_features),
    text=None,
))
fig2.update_layout(
    height=300,
    title="Top 8 Biosignal Feature Importances",
    yaxis_title="Importance",
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

# ── Stats panel ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Performance Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Accuracy", "65.3%", "+8.4% vs baseline")
c2.metric("Biosignal Only", "63.1%", "+6.2% vs baseline")
c3.metric("Landmarks Only", "61.4%", "+4.5% vs baseline")
c4.metric("Evaluation", "LOSO CV")
st.caption("Subject-independent Leave-One-Subject-Out cross-validation on 67 reactive subjects")

st.markdown("---")
st.caption("EmPath Capstone Project | BioVid Heat Pain Database | "
           "Stacked Fusion: RF (Bio) + RF (Landmarks) → Logistic Regression")