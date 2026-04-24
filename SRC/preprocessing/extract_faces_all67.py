import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SAMPLES_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/starting_point/samples.csv"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/faces_all67"

EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames(video_path, num_frames=8):
    """Reduced to 8 frames for speed."""
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(total_frames * 0.40)
    indices      = [int(start_frame + i * (total_frames - start_frame) / num_frames)
                    for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return frames

print("Loading samples...")
samples_df = pd.read_csv(SAMPLES_CSV, sep="\t")
pa_samples = samples_df[samples_df["class_name"].isin(["PA2", "PA3"])]
pa_samples = pa_samples[~pa_samples["subject_name"].isin(EXCLUDED_SUBJECTS)]
print(f"Subjects: {pa_samples['subject_id'].nunique()}")
print(f"Samples:  {len(pa_samples)}")

subjects  = pa_samples["subject_id"].values
gss       = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(pa_samples, groups=subjects))

train_df = pa_samples.iloc[train_idx]
test_df  = pa_samples.iloc[test_idx]
val_size = int(len(train_df) * 0.1)
val_df   = train_df.iloc[:val_size]
train_df = train_df.iloc[val_size:]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def process_split(df, split_name):
    saved   = 0
    failed  = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        video_path = os.path.join(
            BIOVID_ROOT, "video",
            row["subject_name"],
            row["sample_name"] + ".mp4"
        )
        if not os.path.exists(video_path):
            missing += 1
            continue

        out_dir = os.path.join(
            OUTPUT_DIR, split_name, row["class_name"])
        os.makedirs(out_dir, exist_ok=True)

        frames = extract_frames(video_path, num_frames=8)
        if not frames:
            failed += 1
            continue

        for i, frame in enumerate(frames):
            fname = f"{row['sample_name']}_frame{i}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1

    print(f"{split_name}: {saved} frames, {failed} failed, {missing} missing")

process_split(train_df, "train")
process_split(val_df,   "val")
process_split(test_df,  "test")

print("\nFace extraction complete ✓")
print(f"Saved to {OUTPUT_DIR}")