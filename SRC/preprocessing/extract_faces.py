import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SPLITS_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/splits"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/faces_v4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_frame_diff(cap, idx1, idx2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx1)
    ret1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
    ret2, f2 = cap.read()
    if not ret1 or not ret2:
        return 0
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    return np.mean(np.abs(g1.astype(float) - g2.astype(float)))

def extract_smart_frames(video_path, num_frames=24):
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use last 60% of video — pain response window
    start_frame = int(total_frames * 0.40)
    indices     = [int(start_frame + i * (total_frames - start_frame) / num_frames)
                   for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append((idx, frame))
    cap.release()
    return frames

def process_split(split_name):
    split_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df         = pd.read_csv(split_path, sep="\t")
    df         = df[df["class_name"].isin(["PA2", "PA3"])]

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

        frames = extract_smart_frames(video_path)
        if not frames:
            failed += 1
            continue

        for i, (idx, frame) in enumerate(frames):
            fname = f"{row['sample_name']}_frame{i}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1

    print(f"{split_name}: {saved} frames saved, "
          f"{failed} failed, {missing} missing")

for split in ["train", "val", "test"]:
    process_split(split)

print("Face extraction complete ✓")