import pandas as pd
import numpy as np
import os

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SPLITS_DIR = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Experiments/splits"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(bio_path):
    df = pd.read_csv(bio_path, sep="\t")

    gsr = df["gsr"].values
    ecg = df["ecg"].values
    emg_trap = df["emg_trapezius"].values
    emg_corr = df["emg_corrugator"].values
    emg_zyg = df["emg_zygomaticus"].values

    features = {
        "gsr_mean": np.mean(gsr),
        "gsr_std": np.std(gsr),
        "gsr_slope": np.polyfit(np.arange(len(gsr)), gsr, 1)[0],
        "ecg_mean": np.mean(ecg),
        "ecg_std": np.std(ecg),
        "ecg_max": np.max(ecg),
        "emg_trap_mean": np.mean(np.abs(emg_trap)),
        "emg_trap_std": np.std(emg_trap),
        "emg_corr_mean": np.mean(np.abs(emg_corr)),
        "emg_corr_std": np.std(emg_corr),
        "emg_corr_max": np.max(np.abs(emg_corr)),
        "emg_zyg_mean": np.mean(np.abs(emg_zyg)),
        "emg_zyg_std": np.std(emg_zyg),
    }
    return features


def process_split(split_name):
    split_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df_split = pd.read_csv(split_path, sep="\t")

    # Filter only PA2 and PA3
    df_split = df_split[df_split["class_name"].isin(["PA2", "PA3"])]

    rows = []
    failed = 0

    for _, row in df_split.iterrows():
        bio_path = os.path.join(
            BIOVID_ROOT, "biosignals_filtered",
            row["subject_name"],
            row["sample_name"] + "_bio.csv"
        )

        if not os.path.exists(bio_path):
            failed += 1
            continue

        features = extract_features(bio_path)

        # Add metadata columns
        features["subject_id"] = row["subject_id"]
        features["sample_name"] = row["sample_name"]
        features["class_name"] = row["class_name"]
        features["label"] = 0 if row["class_name"] == "PA2" else 1

        rows.append(features)

    output_df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, f"biosignals_{split_name}.csv")
    output_df.to_csv(output_path, index=False)

    print(f"{split_name}: {len(rows)} samples processed, {failed} failed")
    print(f"Saved to {output_path}\n")


# Run for all three splits
for split in ["train", "val", "test"]:
    process_split(split)