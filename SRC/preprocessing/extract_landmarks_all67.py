import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SAMPLES_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/starting_point/samples.csv"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH      = [61, 291, 0, 17, 78, 308]
NOSE       = [1, 2, 98, 327]

def compute_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def extract_landmark_features(frame):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    h, w = frame.shape[:2]
    lm   = results.multi_face_landmarks[0].landmark
    pts  = {i: (lm[i].x * w, lm[i].y * h) for i in range(len(lm))}

    features = {}
    left_brow_center  = np.mean([pts[i] for i in LEFT_BROW],  axis=0)
    right_brow_center = np.mean([pts[i] for i in RIGHT_BROW], axis=0)
    left_eye_center   = np.mean([pts[i] for i in LEFT_EYE],   axis=0)
    right_eye_center  = np.mean([pts[i] for i in RIGHT_EYE],  axis=0)

    features["left_brow_eye_dist"]  = compute_distance(left_brow_center, left_eye_center)
    features["right_brow_eye_dist"] = compute_distance(right_brow_center, right_eye_center)
    features["brow_eye_avg"]        = (features["left_brow_eye_dist"] +
                                       features["right_brow_eye_dist"]) / 2
    features["brow_furrow"]         = compute_distance(pts[LEFT_BROW[0]], pts[RIGHT_BROW[0]])

    left_eye_open  = compute_distance(pts[LEFT_EYE[1]],  pts[LEFT_EYE[5]])
    right_eye_open = compute_distance(pts[RIGHT_EYE[1]], pts[RIGHT_EYE[5]])
    features["left_eye_openness"]  = left_eye_open
    features["right_eye_openness"] = right_eye_open
    features["avg_eye_openness"]   = (left_eye_open + right_eye_open) / 2

    mouth_width  = compute_distance(pts[MOUTH[0]], pts[MOUTH[1]])
    mouth_height = compute_distance(pts[MOUTH[2]], pts[MOUTH[3]])
    features["mouth_width"]        = mouth_width
    features["mouth_height"]       = mouth_height
    features["mouth_aspect_ratio"] = mouth_height / (mouth_width + 1e-6)
    features["nose_width"]         = compute_distance(pts[NOSE[2]], pts[NOSE[3]])

    face_width = compute_distance(pts[234], pts[454])
    for k in list(features.keys()):
        features[k] = features[k] / (face_width + 1e-6)
    return features

def extract_video_features(video_path, num_frames=24):
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(total_frames * 0.40)
    indices      = [int(start_frame + i * (total_frames - start_frame) / num_frames)
                    for i in range(num_frames)]
    all_features = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        feats = extract_landmark_features(frame)
        if feats is not None:
            all_features.append(feats)
    cap.release()

    if len(all_features) == 0:
        return None

    keys   = list(all_features[0].keys())
    result = {}
    for k in keys:
        vals = [f[k] for f in all_features]
        result[f"{k}_mean"] = np.mean(vals)
        result[f"{k}_std"]  = np.std(vals)
    return result

print("Loading samples...")
samples_df = pd.read_csv(SAMPLES_CSV, sep="\t")
pa_samples = samples_df[samples_df["class_name"].isin(["PA2", "PA3"])]
pa_samples = pa_samples[~pa_samples["subject_name"].isin(EXCLUDED_SUBJECTS)]
print(f"Subjects: {pa_samples['subject_id'].nunique()}")
print(f"Samples:  {len(pa_samples)}")

rows    = []
failed  = 0
missing = 0

for _, row in tqdm(pa_samples.iterrows(), total=len(pa_samples)):
    video_path = os.path.join(
        BIOVID_ROOT, "video",
        row["subject_name"],
        row["sample_name"] + ".mp4"
    )
    if not os.path.exists(video_path):
        missing += 1
        continue

    feats = extract_video_features(video_path)
    if feats is None:
        failed += 1
        continue

    feats["subject_id"]  = row["subject_id"]
    feats["sample_name"] = row["sample_name"]
    feats["class_name"]  = row["class_name"]
    feats["label"]       = 0 if row["class_name"] == "PA2" else 1
    rows.append(feats)

output_df = pd.DataFrame(rows)
output_df.to_csv(os.path.join(OUTPUT_DIR, "landmarks_all67.csv"), index=False)

print(f"\nExtracted: {len(rows)} samples")
print(f"Failed:    {failed}")
print(f"Missing:   {missing}")
print(f"Saved ✓")