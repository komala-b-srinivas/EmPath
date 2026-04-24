"""
Extract raw MediaPipe FaceMesh landmark coordinates for the GNN experiment.

Unlike extract_landmarks_all67.py (which computes 22 hand-engineered distance features),
this script stores the raw (x, y) position of all 468 landmarks per frame.
The GNN will learn which spatial relationships matter directly from the data.

Uses MediaPipe 0.10.x Tasks API (FaceLandmarker) — compatible with M3 Apple Silicon.
Downloads the FaceLandmarker model (~30 MB) on first run if not already present.

Output:
    Results/landmarks_gnn/raw_coords.npz
    ├── coords       : float32 (N, 24, 468, 2)  — normalized [0,1] x/y per frame
    ├── labels       : int64   (N,)              — 0=PA2, 1=PA3
    ├── subject_ids  : int64   (N,)
    └── sample_names : str     (N,)

N ≈ 2680 samples, 24 frames per sample, 468 landmarks, (x, y) per landmark.
Uncompressed size ≈ 240 MB. Saved as compressed .npz (expect ~60–80 MB on disk).

Usage:
    python SRC/preprocessing/extract_landmarks_raw_coords.py

Runtime: ~45–60 min on M3 CPU.
"""

import cv2
import numpy as np
import pandas as pd
import os
import urllib.request
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIOVID_ROOT = os.path.join(BASE_DIR, "Data", "Raw")
SAMPLES_CSV = os.path.join(BASE_DIR, "Data", "Raw", "starting_point", "samples.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Results", "landmarks_gnn")
MODEL_DIR   = os.path.join(BASE_DIR, "Models")
MODEL_PATH  = os.path.join(MODEL_DIR, "face_landmarker.task")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_FRAMES    = 24   # frames sampled per video (last 60% = pain response window)
NUM_LANDMARKS = 468  # standard FaceMesh landmarks (FaceLandmarker returns 478 incl. iris)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}

# ── Download model if needed ───────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"Downloading FaceLandmarker model (~30 MB) → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  Download complete.")
else:
    print(f"Model found: {MODEL_PATH}")

# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
_base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options   = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    num_faces=1,
    min_face_detection_confidence=0.5,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(_options)


def extract_frame_coords(frame):
    """
    Run FaceLandmarker on a single BGR frame.

    Returns:
        np.ndarray of shape (468, 2) with normalized (x, y) in [0, 1],
        or None if no face is detected.
    """
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm_list = result.face_landmarks[0]          # list of NormalizedLandmark
    # FaceLandmarker returns 478 points (468 face + 10 iris); take first 468
    coords = np.array(
        [[lm.x, lm.y] for lm in lm_list[:NUM_LANDMARKS]],
        dtype=np.float32
    )
    return coords


def extract_video_coords(video_path, num_frames=NUM_FRAMES):
    """
    Sample `num_frames` frames from a video (last 60% = pain response window)
    and extract landmark coordinates for each.

    Returns:
        np.ndarray of shape (num_frames, 468, 2), failed frames filled with
        mean of successful frames. Returns None if all frames fail.
    """
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(total_frames * 0.40)

    indices = [
        int(start_frame + i * (total_frames - start_frame) / num_frames)
        for i in range(num_frames)
    ]

    frame_coords = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame_coords.append(None)
            continue
        frame_coords.append(extract_frame_coords(frame))

    cap.release()

    valid = [c for c in frame_coords if c is not None]
    if len(valid) == 0:
        return None

    fill_value   = np.mean(valid, axis=0)
    frame_coords = [c if c is not None else fill_value for c in frame_coords]
    return np.stack(frame_coords, axis=0)   # (24, 468, 2)


# ── Main extraction loop ───────────────────────────────────────────────────────
print("\nLoading samples CSV...")
samples_df = pd.read_csv(SAMPLES_CSV, sep="\t")
pa_samples = samples_df[samples_df["class_name"].isin(["PA2", "PA3"])]
pa_samples = pa_samples[~pa_samples["subject_name"].isin(EXCLUDED_SUBJECTS)]
print(f"  Subjects : {pa_samples['subject_id'].nunique()}")
print(f"  Samples  : {len(pa_samples)}")

all_coords       = []
all_labels       = []
all_subject_ids  = []
all_sample_names = []
n_failed  = 0
n_missing = 0

for _, row in tqdm(pa_samples.iterrows(), total=len(pa_samples), desc="Extracting"):
    video_path = os.path.join(
        BIOVID_ROOT, "video",
        row["subject_name"],
        row["sample_name"] + ".mp4"
    )

    if not os.path.exists(video_path):
        n_missing += 1
        continue

    coords = extract_video_coords(video_path)
    if coords is None:
        n_failed += 1
        continue

    all_coords.append(coords)
    all_labels.append(0 if row["class_name"] == "PA2" else 1)
    all_subject_ids.append(int(row["subject_id"]))
    all_sample_names.append(row["sample_name"])

print(f"\nResults:")
print(f"  Extracted : {len(all_coords)} samples")
print(f"  Failed    : {n_failed}  (no face detected in any frame)")
print(f"  Missing   : {n_missing}  (video file not found)")

coords_array       = np.stack(all_coords, axis=0).astype(np.float32)
labels_array       = np.array(all_labels, dtype=np.int64)
subject_ids_array  = np.array(all_subject_ids, dtype=np.int64)
sample_names_array = np.array(all_sample_names)

print(f"\nCoords array shape : {coords_array.shape}")
print(f"  dtype            : {coords_array.dtype}")
print(f"  uncompressed     : {coords_array.nbytes / 1e6:.0f} MB")
print(f"  x range          : [{coords_array[...,0].min():.3f}, {coords_array[...,0].max():.3f}]")
print(f"  y range          : [{coords_array[...,1].min():.3f}, {coords_array[...,1].max():.3f}]")

out_path = os.path.join(OUTPUT_DIR, "raw_coords.npz")
np.savez_compressed(
    out_path,
    coords       = coords_array,
    labels       = labels_array,
    subject_ids  = subject_ids_array,
    sample_names = sample_names_array
)
print(f"\nSaved: {out_path}")
print(f"  On-disk size: ~{os.path.getsize(out_path) / 1e6:.0f} MB")
print("\nDone. Run evaluate_gnn_landmarks_loso.py next.")
