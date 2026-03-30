import pandas as pd
import numpy as np
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67/landmarks_all67.csv"
SAMPLES_CSV   = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/starting_point/samples.csv"

EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}

print("Loading landmark data...")
all_df = pd.read_csv(LANDMARKS_CSV)
print(f"Total samples: {len(all_df)}")

samples_df  = pd.read_csv(SAMPLES_CSV, sep="\t")
excluded_ids = samples_df[
    samples_df["subject_name"].isin(EXCLUDED_SUBJECTS)
]["subject_id"].unique()

all_df = all_df[~all_df["subject_id"].isin(excluded_ids)]
print(f"After excluding non-reactive: {len(all_df)} samples, "
      f"{all_df['subject_id'].nunique()} subjects")

EXCLUDE  = ["subject_id", "sample_name", "class_name", "label"]
lm_cols  = [c for c in all_df.columns if c not in EXCLUDE]

X      = np.nan_to_num(all_df[lm_cols].values)
y      = all_df["label"].values
groups = all_df["subject_id"].values

print(f"Features: {len(lm_cols)}")

logo = LeaveOneGroupOut()
accs = []
fold = 0

print("\nRunning LOSO with person-specific normalization...")
for train_idx, test_idx in logo.split(X, y, groups):
    fold += 1
    X_train_raw  = X[train_idx]
    X_test_raw   = X[test_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]
    groups_train = groups[train_idx]

    X_train_norm = X_train_raw.copy()
    for subj_id in np.unique(groups_train):
        mask = groups_train == subj_id
        mean = X_train_raw[mask].mean(axis=0)
        std  = X_train_raw[mask].std(axis=0)
        std[std == 0] = 1
        X_train_norm[mask] = (X_train_raw[mask] - mean) / std

    test_mean = X_test_raw.mean(axis=0)
    test_std  = X_test_raw.std(axis=0)
    test_std[test_std == 0] = 1
    X_test_norm = (X_test_raw - test_mean) / test_std

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
print(f"  VISION LOSO RESULTS (Landmarks, 67 subjects)")
print(f"{'='*55}")
print(f"  Previous (random split) : 51.6%")
print(f"  LOSO + person-norm      : {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
print(f"  Biosignal LOSO (best)   : 63.1% ± 11.6%")
print(f"{'='*55}")