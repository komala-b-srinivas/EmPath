import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

BIOSIG_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/biosignals_filtered"
DEMO_PATH   = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Models/demo_samples.csv"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Models/signal_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

samples = pd.read_csv(DEMO_PATH)
print(f"Generating plots for {len(samples)} samples...")

failed = 0
saved  = 0

for _, row in samples.iterrows():
    sname   = row["sample_name"]
    subject = sname.split("-")[0]
    path    = os.path.join(BIOSIG_DIR, subject, sname + "_bio.csv")

    if not os.path.exists(path):
        failed += 1
        continue

    try:
        df   = pd.read_csv(path, sep="\t")
        time = np.arange(len(df)) / 512

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("GSR (Skin Conductance)",
                            "ECG (Heart Rate)",
                            "EMG Corrugator (Facial Muscle)"),
            vertical_spacing=0.12
        )
        fig.add_trace(go.Scatter(x=time, y=df["gsr"],
            line=dict(color="#1976d2", width=1.5), name="GSR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=df["ecg"],
            line=dict(color="#d32f2f", width=1), name="ECG"), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=df["emg_corrugator"],
            line=dict(color="#388e3c", width=1), name="EMG"), row=3, col=1)
        fig.update_layout(
            height=400, showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white")
        )
        fig.update_xaxes(title_text="Time (s)", row=3, col=1,
                         gridcolor="#333", color="white")
        fig.update_yaxes(gridcolor="#333", color="white")

        out_path = os.path.join(OUTPUT_DIR, sname + ".png")
        fig.write_image(out_path, width=700, height=400)
        saved += 1

        if saved % 50 == 0:
            print(f"  Saved {saved} plots...")

    except Exception as e:
        failed += 1
        print(f"  Failed {sname}: {e}")

print(f"\nDone: {saved} saved, {failed} failed")
print(f"Output: {OUTPUT_DIR}")