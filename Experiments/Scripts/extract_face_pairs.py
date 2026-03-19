import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SPLITS_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Experiments/splits"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/faces_v3"

NUM_FRAMES = 5
FACE_SIZE  = (224, 224)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_frame_pairs(video_path, num_frames=5):
    """
    Returns pairs of (current_frame, prev_frame) from
    the most expressive window.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_frames  = []
    gray_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(all_frames) < 2:
        return []

    # Compute diff scores
    diff_scores = [0]
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff_scores.append(np.mean(diff))
    diff_scores = np.array(diff_scores)

    # Find most active window
    window_size = int(fps * 1.5)
    best_start  = 1  # start at 1 so we always have a previous frame
    best_score  = 0

    for i in range(1, len(diff_scores) - window_size):
        window_score = np.mean(diff_scores[i:i + window_size])
        if window_score > best_score:
            best_score = window_score
            best_start = i

    best_end = min(best_start + window_size, len(all_frames))

    # Sample frame pairs from this window
    indices = [int(best_start + i * (best_end - best_start) / num_frames)
               for i in range(num_frames)]

    pairs = []
    for idx in indices:
        if idx > 0 and idx < len(all_frames):
            pairs.append((all_frames[idx], all_frames[idx-1]))

    return pairs


def crop_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE)
    return face


def process_split(split_name):
    split_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df = pd.read_csv(split_path, sep="\t")

    total_saved   = 0
    total_failed  = 0
    total_missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        video_path = os.path.join(
            BIOVID_ROOT, "video",
            row["subject_name"],
            row["sample_name"] + ".mp4"
        )

        if not os.path.exists(video_path):
            total_missing += 1
            continue

        pairs = extract_frame_pairs(video_path, NUM_FRAMES)

        for i, (curr_frame, prev_frame) in enumerate(pairs):
            curr_face = crop_face(curr_frame)
            prev_face = crop_face(prev_frame)

            if curr_face is None or prev_face is None:
                total_failed += 1
                continue

            # Compute difference image
            diff_face = cv2.absdiff(curr_face, prev_face)

            # Save all three
            save_dir = os.path.join(OUTPUT_DIR, split_name,
                                    row["class_name"], row["sample_name"])
            os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, f"curr_{i}.jpg"),  curr_face)
            cv2.imwrite(os.path.join(save_dir, f"prev_{i}.jpg"),  prev_face)
            cv2.imwrite(os.path.join(save_dir, f"diff_{i}.jpg"),  diff_face)
            total_saved += 1

    print(f"\n{split_name} done:")
    print(f"  Saved        : {total_saved} frame pairs")
    print(f"  No face      : {total_failed} skipped")
    print(f"  Missing video: {total_missing} not found\n")


for split in ["train", "val", "test"]:
    process_split(split)

print("All done! Face pair extraction complete ✓")