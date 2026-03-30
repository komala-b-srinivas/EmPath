import pandas as pd
import numpy as np
import os
from tqdm import tqdm

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SPLITS_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/splits"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(bio_path):
    df   = pd.read_csv(bio_path, sep="\t")
    gsr  = df["gsr"].values
    ecg  = df["ecg"].values
    trap = df["emg_trapezius"].values
    corr = df["emg_corrugator"].values
    zyg  = df["emg_zygomaticus"].values

    return {
        "gsr_mean":      np.mean(gsr),
        "gsr_std":       np.std(gsr),
        "gsr_slope":     np.polyfit(np.arange(len(gsr)), gsr, 1)[0],
        "ecg_mean":      np.mean(ecg),
        "ecg_std":       np.std(ecg),
        "ecg_max":       np.max(ecg),
        "emg_trap_mean": np.mean(np.abs(trap)),
        "emg_trap_std":  np.std(trap),
        "emg_corr_mean": np.mean(np.abs(corr)),
        "emg_corr_std":  np.std(corr),
        "emg_corr_max":  np.max(np.abs(corr)),
        "emg_zyg_mean":  np.mean(np.abs(zyg)),
        "emg_zyg_std":   np.std(zyg),
    }

def process_split(split_name):
    split_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df         = pd.read_csv(split_path, sep="\t")
    df         = df[df["class_name"].isin(["PA2", "PA3"])]

    rows   = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        bio_path = os.path.join(
            BIOVID_ROOT, "biosignals_filtered",
            row["subject_name"],
            row["sample_name"] + "_bio.csv"
        )
        if not os.path.exists(bio_path):
            failed += 1
            continue

        feats = extract_features(bio_path)
        feats["subject_id"]  = row["subject_id"]
        feats["sample_name"] = row["sample_name"]
        feats["class_name"]  = row["class_name"]
        feats["label"]       = 0 if row["class_name"] == "PA2" else 1
        rows.append(feats)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(
        os.path.join(OUTPUT_DIR, f"biosignals_{split_name}.csv"),
        index=False)

    print(f"{split_name}: {len(rows)} samples, {failed} failed")

for split in ["train", "val", "test"]:
    process_split(split)

print("Done ✓")