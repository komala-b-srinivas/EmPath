import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67/landmarks_all67.csv"
BIOSIG_CSV    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals_hrv/all_67_hrv.csv"

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
        "subject_id": lm_row["subject_id"],
        "label":      lm_row["label"],
        "combined":   np.concatenate([bio_feats, lm_feats])
    })

X      = np.array([r["combined"] for r in rows])
y      = np.array([r["label"]    for r in rows])
groups = np.array([r["subject_id"] for r in rows])

print(f"Matched: {len(rows)} samples")
print(f"Combined features: {X.shape[1]}")

logo = LeaveOneGroupOut()
accs = []
fold = 0

def person_norm_train(X, g):
    X_norm = X.copy()
    for sid in np.unique(g):
        mask = g == sid
        mean = X[mask].mean(axis=0)
        std  = X[mask].std(axis=0)
        std[std == 0] = 1
        X_norm[mask] = (X[mask] - mean) / std
    return X_norm

def person_norm_test(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std

print("\nRunning Early Fusion LOSO...")
for train_idx, test_idx in logo.split(X, y, groups):
    fold += 1
    groups_train = groups[train_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]

    X_train_norm = person_norm_train(X[train_idx], groups_train)
    X_test_norm  = person_norm_test(X[test_idx])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=4,
        min_samples_split=10, max_features='sqrt',
        random_state=42, n_jobs=-1)
    rf.fit(X_train_norm, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test_norm))
    accs.append(acc)

    if fold % 10 == 0:
        print(f"  Fold {fold} | Running mean: {np.mean(accs)*100:.1f}%")

print(f"\n{'='*55}")
print(f"  EARLY FUSION LOSO RESULTS")
print(f"{'='*55}")
print(f"  Biosignal only  : 63.1% ± 11.6%")
print(f"  Landmarks only  : 61.4% ± 13.1%")
print(f"  Early fusion    : {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
print(f"{'='*55}")