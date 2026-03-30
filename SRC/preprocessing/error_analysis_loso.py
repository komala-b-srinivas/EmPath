import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

LANDMARKS_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/landmarks_all67/landmarks_all67.csv"
BIOSIG_CSV    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals_hrv/all_67_hrv.csv"
OUTPUT_DIR    = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        "label":       lm_row["label"],
        "class_name":  lm_row["class_name"],
        "bio":         bio_feats,
        "lm":          lm_feats,
    })

X_bio  = np.array([r["bio"] for r in rows])
X_lm   = np.array([r["lm"]  for r in rows])
y      = np.array([r["label"] for r in rows])
groups = np.array([r["subject_id"] for r in rows])
names  = [r["sample_name"] for r in rows]

print(f"Matched: {len(rows)} samples")

logo = LeaveOneGroupOut()

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

print("Running stacked fusion LOSO...")
all_preds    = np.zeros(len(rows), dtype=int)
all_labels   = np.zeros(len(rows), dtype=int)
all_probs    = np.zeros(len(rows))
subject_accs = {}

for train_idx, test_idx in logo.split(X_bio, y, groups):
    groups_train = groups[train_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]
    subj_id      = groups[test_idx[0]]

    X_bio_train = person_norm_train(X_bio[train_idx], groups_train)
    X_bio_test  = person_norm_test(X_bio[test_idx])
    X_lm_train  = person_norm_train(X_lm[train_idx],  groups_train)
    X_lm_test   = person_norm_test(X_lm[test_idx])

    rf_bio = RandomForestClassifier(n_estimators=300, max_depth=4,
                                     min_samples_split=10, max_features='sqrt',
                                     random_state=42, n_jobs=-1)
    rf_bio.fit(X_bio_train, y_train)

    rf_lm = RandomForestClassifier(n_estimators=300, max_depth=4,
                                    min_samples_split=10, max_features='sqrt',
                                    random_state=42, n_jobs=-1)
    rf_lm.fit(X_lm_train, y_train)

    bio_train_probs = rf_bio.predict_proba(X_bio_train)
    lm_train_probs  = rf_lm.predict_proba(X_lm_train)
    bio_test_probs  = rf_bio.predict_proba(X_bio_test)
    lm_test_probs   = rf_lm.predict_proba(X_lm_test)

    X_meta_train = np.hstack([bio_train_probs, lm_train_probs])
    X_meta_test  = np.hstack([bio_test_probs,  lm_test_probs])

    meta = LogisticRegression(random_state=42, max_iter=1000)
    meta.fit(X_meta_train, y_train)

    preds = meta.predict(X_meta_test)
    probs = meta.predict_proba(X_meta_test)[:, 1]

    all_preds[test_idx]  = preds
    all_labels[test_idx] = y_test
    all_probs[test_idx]  = probs
    subject_accs[subj_id] = accuracy_score(y_test, preds)

print("Done ✓")

# ── Overall results ────────────────────────────────────────────────────
print(f"\nAccuracy: {accuracy_score(all_labels, all_preds)*100:.1f}%")
print(classification_report(all_labels, all_preds,
                             target_names=["PA2", "PA3"]))

# ── Confusion matrix ───────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["PA2", "PA3"],
            yticklabels=["PA2", "PA3"])
plt.title("EmPath Stacked Fusion — Confusion Matrix\nLOSO 67 Subjects")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_final.png"), dpi=150)
plt.close()

# ── Per-subject accuracy ───────────────────────────────────────────────
subject_df = pd.DataFrame([
    {"subject_id": k, "accuracy": v}
    for k, v in subject_accs.items()
]).sort_values("accuracy")

plt.figure(figsize=(14, 5))
colors = ["#d32f2f" if a < 0.5 else "#388e3c" if a >= 0.65 else "#f57c00"
          for a in subject_df["accuracy"]]
plt.bar(range(len(subject_df)), subject_df["accuracy"], color=colors)
plt.axhline(y=0.5,   color="red",  linestyle="--", alpha=0.7, label="Chance (50%)")
plt.axhline(y=0.653, color="blue", linestyle="--", alpha=0.7, label="Mean (65.3%)")
plt.xlabel("Subject (sorted by accuracy)")
plt.ylabel("Accuracy")
plt.title("Per-Subject Accuracy — EmPath Stacked Fusion")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_subject_accuracy.png"), dpi=150)
plt.close()

print(f"Best subjects  (>80%): {sum(1 for a in subject_accs.values() if a > 0.8)}")
print(f"Good subjects  (65-80%): {sum(1 for a in subject_accs.values() if 0.65 <= a <= 0.8)}")
print(f"Chance subjects(50-65%): {sum(1 for a in subject_accs.values() if 0.5 <= a < 0.65)}")
print(f"Below chance   (<50%): {sum(1 for a in subject_accs.values() if a < 0.5)}")

# ── Feature importance ─────────────────────────────────────────────────
all_feature_names = bio_cols + lm_cols
importances = []

for train_idx, _ in logo.split(np.hstack([X_bio, X_lm]), y, groups):
    X_combined   = np.hstack([X_bio, X_lm])
    groups_train = groups[train_idx]
    X_train      = person_norm_train(X_combined[train_idx], groups_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                 min_samples_split=10, max_features='sqrt',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y[train_idx])
    importances.append(rf.feature_importances_)

mean_importance = np.mean(importances, axis=0)
importance_df   = pd.DataFrame({
    "feature":    all_feature_names,
    "importance": mean_importance
}).sort_values("importance", ascending=False)

print("\nTop 15 features:")
print(importance_df.head(15).to_string())

plt.figure(figsize=(14, 6))
top15  = importance_df.head(15)
colors = ["#1976d2" if f in bio_cols else "#388e3c"
          for f in top15["feature"]]
plt.bar(range(15), top15["importance"], color=colors)
plt.xticks(range(15), top15["feature"], rotation=45, ha="right", fontsize=9)
plt.title("Top 15 Feature Importances\nBlue=Biosignal, Green=Landmark")
plt.ylabel("Mean Importance (LOSO)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

subject_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_accuracy.csv"), index=False)
print(f"\nAll saved to {OUTPUT_DIR} ✓")