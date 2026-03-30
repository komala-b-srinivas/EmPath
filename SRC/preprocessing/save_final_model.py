import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67/landmarks_all67.csv"
BIOSIG_CSV    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals_hrv/all_67_hrv.csv"
MODEL_DIR     = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
lm_df  = pd.read_csv(LANDMARKS_CSV)
bio_df = pd.read_csv(BIOSIG_CSV)

EXCLUDE  = ["subject_id", "sample_name", "class_name", "label"]
lm_cols  = [c for c in lm_df.columns  if c not in EXCLUDE]
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]

bio_dict = {row["sample_name"]: row for _, row in bio_df.iterrows()}

rows = []
for _, lm_row in lm_df.iterrows():
    sname = lm_row["sample_name"]
    if sname not in bio_dict:
        continue
    bio_row   = bio_dict[sname]
    lm_feats  = np.nan_to_num(lm_row[lm_cols].values.astype(float))
    bio_feats = np.nan_to_num(bio_row[bio_cols].values.astype(float))
    rows.append({
        "subject_id":  lm_row["subject_id"],
        "sample_name": sname,
        "class_name":  lm_row["class_name"],
        "label":       lm_row["label"],
        "bio":         bio_feats,
        "lm":          lm_feats,
    })

X_bio  = np.array([r["bio"] for r in rows])
X_lm   = np.array([r["lm"]  for r in rows])
y      = np.array([r["label"] for r in rows])
groups = np.array([r["subject_id"] for r in rows])

print(f"Total samples: {len(rows)}")

def person_norm_train(X, g):
    X_norm = X.copy()
    stats  = {}
    for sid in np.unique(g):
        mask = g == sid
        mean = X[mask].mean(axis=0)
        std  = X[mask].std(axis=0)
        std[std == 0] = 1
        X_norm[mask]  = (X[mask] - mean) / std
        stats[sid]    = {"mean": mean, "std": std}
    return X_norm, stats

print("Training final model on all 67 subjects...")
X_bio_norm, _ = person_norm_train(X_bio, groups)
X_lm_norm,  _ = person_norm_train(X_lm,  groups)

global_bio_mean = X_bio.mean(axis=0)
global_bio_std  = X_bio.std(axis=0)
global_bio_std[global_bio_std == 0] = 1
global_lm_mean  = X_lm.mean(axis=0)
global_lm_std   = X_lm.std(axis=0)
global_lm_std[global_lm_std == 0] = 1

rf_bio = RandomForestClassifier(
    n_estimators=300, max_depth=4,
    min_samples_split=10, max_features='sqrt',
    random_state=42, n_jobs=-1)
rf_bio.fit(X_bio_norm, y)

rf_lm = RandomForestClassifier(
    n_estimators=300, max_depth=4,
    min_samples_split=10, max_features='sqrt',
    random_state=42, n_jobs=-1)
rf_lm.fit(X_lm_norm, y)

bio_probs = rf_bio.predict_proba(X_bio_norm)
lm_probs  = rf_lm.predict_proba(X_lm_norm)
X_meta    = np.hstack([bio_probs, lm_probs])

meta = LogisticRegression(random_state=42, max_iter=1000)
meta.fit(X_meta, y)

train_preds = meta.predict(X_meta)
print(f"Training accuracy: {accuracy_score(y, train_preds)*100:.1f}%")

model_package = {
    "rf_bio":          rf_bio,
    "rf_lm":           rf_lm,
    "meta":            meta,
    "bio_cols":        bio_cols,
    "lm_cols":         lm_cols,
    "global_bio_mean": global_bio_mean,
    "global_bio_std":  global_bio_std,
    "global_lm_mean":  global_lm_mean,
    "global_lm_std":   global_lm_std,
    "loso_accuracy":   0.653,
}

with open(os.path.join(MODEL_DIR, "empath_model.pkl"), "wb") as f:
    pickle.dump(model_package, f)

sample_data = []
for r in rows:
    sample_data.append({
        "subject_id":  r["subject_id"],
        "sample_name": r["sample_name"],
        "class_name":  r["class_name"],
        "label":       r["label"],
        **{f"bio_{i}": v for i, v in enumerate(r["bio"])},
        **{f"lm_{i}":  v for i, v in enumerate(r["lm"])},
    })

pd.DataFrame(sample_data).to_csv(
    os.path.join(MODEL_DIR, "demo_samples.csv"), index=False)

print(f"Model saved ✓")
print(f"Demo samples saved ✓")