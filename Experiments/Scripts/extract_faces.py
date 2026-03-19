import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SPLITS_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Experiments/splits"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/faces_v2"

NUM_FRAMES  = 10
FACE_SIZE   = (224, 224)
# ──────────────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_frames(video_path, num_frames=5):
    """
    Automatically finds the most expressive window in the video
    using frame difference detection, then samples frames from it.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames
    all_frames  = []
    gray_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(all_frames) == 0:
        return []

    # Compute frame difference scores
    diff_scores = [0]
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff_scores.append(np.mean(diff))
    diff_scores = np.array(diff_scores)

    # Find most active 1.5 second window
    window_size = int(fps * 1.5)
    best_start  = 0
    best_score  = 0

    for i in range(len(diff_scores) - window_size):
        window_score = np.mean(diff_scores[i:i + window_size])
        if window_score > best_score:
            best_score   = window_score
            best_start   = i

    best_end = min(best_start + window_size, len(all_frames))

    # Sample num_frames evenly from this window
    indices = [int(best_start + i * (best_end - best_start) / num_frames)
               for i in range(num_frames)]

    return [all_frames[i] for i in indices if i < len(all_frames)]


def crop_face(frame):
    """Detect and crop the largest face in a frame."""
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

        frames = extract_frames(video_path, NUM_FRAMES)

        for i, frame in enumerate(frames):
            face = crop_face(frame)

            if face is None:
                total_failed += 1
                continue

            # Save to faces_v2/split/class/subject_sample_frameN.jpg
            save_dir = os.path.join(OUTPUT_DIR, split_name, row["class_name"])
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{row['sample_name']}_frame{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), face)
            total_saved += 1

    print(f"\n{split_name} done:")
    print(f"  Saved        : {total_saved} face images")
    print(f"  No face      : {total_failed} frames skipped")
    print(f"  Missing video: {total_missing} files not found\n")


# ── Run for all three splits ───────────────────────────────────────────────
for split in ["train", "val", "test"]:
    process_split(split)

print("All done! Face extraction v2 complete ✓")