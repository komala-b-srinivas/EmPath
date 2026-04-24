"""
SHAP / XAI Analysis — EmPath Stacked Fusion
============================================
Explainability layer for the clinical credibility of EmPath v2.

WHY SHAP:
  The RF stacked fusion is a black box. Clinical stakeholders (ICU, dementia
  care teams) need to know *why* the model predicts PA2 vs PA3 before trusting
  it. SHAP TreeExplainer gives exact Shapley values for each feature's
  contribution to each prediction.

WHAT THIS GENERATES:
  1. SHAP summary plot — biosignal RF: top features ranked by mean |SHAP|
  2. SHAP beeswarm — biosignal RF: feature × value × SHAP magnitude
  3. SHAP summary plot — landmark RF: top landmark distance features
  4. Confusion matrix — stacked fusion (LOSO 67 subjects)
  5. Per-subject accuracy bar chart — identifies hard vs easy subjects
  6. Feature importance bar chart — RF combined importance (for comparison)

METHODOLOGY:
  For each LOSO fold:
    - Train RF_bio, RF_lm on 66 subjects (person-normalized)
    - TreeExplainer(RF_bio) on test subject → collect SHAP values
    - TreeExplainer(RF_lm) on test subject → collect SHAP values
    - Run stacked meta-learner → collect predictions
  Aggregate SHAP values across all 67 test folds → plot global summary.

  Using out-of-fold SHAP ensures no data leakage — every SHAP value comes
  from a subject the model never saw during training.

Usage:
    pip install shap  # if not installed
    python SRC/preprocessing/shap_analysis_loso.py

Runtime: ~5-10 min (67 LOSO folds × TreeExplainer, CPU only).
Output:  Results/error_analysis_v2/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    sys.exit(1)


def _shap_class1(sv) -> np.ndarray:
    """
    Extract SHAP values for class 1 (PA3 / more pain), handling all shap
    versions:
      - Old (<0.41): list of two arrays [class0, class1]
      - Mid (0.41):  3-D ndarray (n_samples, n_features, n_classes)
      - New (0.42+): Explanation object with .values attribute
    For binary RF, newer shap may return (n_samples, n_features) directly.
    """
    if hasattr(sv, "values"):          # Explanation object
        sv = sv.values
    if isinstance(sv, list):
        return np.array(sv[1])         # class-1 array
    sv = np.array(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]             # shape (N, F, 2) → (N, F)
    return sv                          # already (N, F)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIOSIG_CSV  = os.path.join(BASE_DIR, "Results", "biosignals_hrv",  "all_67_hrv.csv")
LM_CSV      = os.path.join(BASE_DIR, "Results", "landmarks_all67", "landmarks_all67.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Results", "error_analysis_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print(f"Output directory : {OUTPUT_DIR}")
print(f"Biosignal CSV    : {BIOSIG_CSV}")
print(f"Landmark CSV     : {LM_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading biosignal features...")
if not os.path.exists(BIOSIG_CSV):
    print(f"ERROR: {BIOSIG_CSV} not found.")
    sys.exit(1)
bio_df   = pd.read_csv(BIOSIG_CSV)
EXCLUDE  = {"subject_id", "sample_name", "class_name", "label"}
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]
print(f"  {len(bio_df)} samples, {len(bio_cols)} biosignal features")

print("Loading landmark features...")
if not os.path.exists(LM_CSV):
    print(f"ERROR: {LM_CSV} not found.")
    sys.exit(1)
lm_df   = pd.read_csv(LM_CSV)
lm_cols = [c for c in lm_df.columns if c not in EXCLUDE]
print(f"  {len(lm_df)} samples, {len(lm_cols)} landmark features")

# Align by sample_name
bio_lu = {row["sample_name"]: row for _, row in bio_df.iterrows()}
lm_lu  = {row["sample_name"]: row for _, row in lm_df.iterrows()}
common = sorted(set(bio_lu) & set(lm_lu))
print(f"  Matched: {len(common)} samples in both modalities")

X_bio    = np.nan_to_num(
    np.array([bio_lu[s][bio_cols].values.astype(np.float32) for s in common])
)
X_lm     = np.nan_to_num(
    np.array([lm_lu[s][lm_cols].values.astype(np.float32) for s in common])
)
y        = np.array([bio_lu[s]["label"]      for s in common], dtype=np.int64)
groups   = np.array([bio_lu[s]["subject_id"] for s in common], dtype=np.int64)
n_subj   = np.unique(groups).size
print(f"  Subjects: {n_subj}  |  Classes: PA2={sum(y==0)}, PA3={sum(y==1)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def person_norm_train(X: np.ndarray, grps: np.ndarray) -> np.ndarray:
    out = X.copy()
    for sid in np.unique(grps):
        mask      = grps == sid
        mean, std = X[mask].mean(0), X[mask].std(0)
        std[std == 0] = 1.0
        out[mask] = (X[mask] - mean) / std
    return out

def person_norm_test(X: np.ndarray) -> np.ndarray:
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    return (X - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSO — collect predictions + SHAP values
# ══════════════════════════════════════════════════════════════════════════════

logo    = LeaveOneGroupOut()
n_folds = n_subj
print(f"\nRunning LOSO ({n_folds} folds) — collecting SHAP values...")
print("=" * 65)

# Storage
all_preds    = np.zeros(len(y), dtype=int)
all_labels   = np.zeros(len(y), dtype=int)
subject_accs = {}

# SHAP accumulators  (out-of-fold test SHAP values, aggregated across folds)
shap_bio_list = []   # list of (n_test, n_bio_features) arrays
shap_lm_list  = []   # list of (n_test, n_lm_features) arrays
X_bio_test_all = []   # matching raw test features (for beeswarm)
X_lm_test_all  = []

fold = 0
for train_idx, test_idx in logo.split(X_bio, y, groups):
    fold += 1
    g_train  = groups[train_idx]
    y_train  = y[train_idx]
    y_test   = y[test_idx]
    subj_id  = groups[test_idx[0]]

    # Normalise
    Xb_tr = person_norm_train(X_bio[train_idx], g_train)
    Xb_te = person_norm_test(X_bio[test_idx])
    Xl_tr = person_norm_train(X_lm[train_idx],  g_train)
    Xl_te = person_norm_test(X_lm[test_idx])

    # ── Train RF models ────────────────────────────────────────────────────
    rf_bio = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf_bio.fit(Xb_tr, y_train)

    rf_lm = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf_lm.fit(Xl_tr, y_train)

    # ── Stacked fusion ─────────────────────────────────────────────────────
    bio_tr_p = rf_bio.predict_proba(Xb_tr)
    lm_tr_p  = rf_lm.predict_proba(Xl_tr)
    bio_te_p = rf_bio.predict_proba(Xb_te)
    lm_te_p  = rf_lm.predict_proba(Xl_te)

    meta = LogisticRegression(max_iter=1000, random_state=SEED)
    meta.fit(np.hstack([bio_tr_p, lm_tr_p]), y_train)
    preds = meta.predict(np.hstack([bio_te_p, lm_te_p]))

    all_preds[test_idx]  = preds
    all_labels[test_idx] = y_test
    subject_accs[subj_id] = accuracy_score(y_test, preds)

    # ── SHAP: TreeExplainer on biosignal RF ───────────────────────────────
    explainer_bio = shap.TreeExplainer(rf_bio)
    sv_bio = _shap_class1(explainer_bio.shap_values(Xb_te))
    shap_bio_list.append(sv_bio)
    X_bio_test_all.append(Xb_te)

    # ── SHAP: TreeExplainer on landmark RF ────────────────────────────────
    explainer_lm = shap.TreeExplainer(rf_lm)
    sv_lm = _shap_class1(explainer_lm.shap_values(Xl_te))
    shap_lm_list.append(sv_lm)
    X_lm_test_all.append(Xl_te)

    acc = subject_accs[subj_id]
    print(f"  Fold {fold:2d}/{n_folds} | subj {subj_id:3d} | "
          f"acc {acc*100:.0f}% | "
          f"running mean {np.mean(list(subject_accs.values()))*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════

shap_bio_all = np.vstack(shap_bio_list)    # (N, n_bio_features)
shap_lm_all  = np.vstack(shap_lm_list)    # (N, n_lm_features)
X_bio_all    = np.vstack(X_bio_test_all)  # (N, n_bio_features) — normalized test vals
X_lm_all     = np.vstack(X_lm_test_all)  # (N, n_lm_features)

overall_acc = accuracy_score(all_labels, all_preds)
print(f"\n{'=' * 65}")
print(f"  Overall LOSO accuracy : {overall_acc*100:.1f}%")
print(f"  (Expected baseline    : 65.3% ± 14.1%)")
print(f"{'=' * 65}")
print(classification_report(all_labels, all_preds, target_names=["PA2", "PA3"]))


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOT 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["PA2 (43°C)", "PA3 (45°C)"],
            yticklabels=["PA2 (43°C)", "PA3 (45°C)"],
            ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"})
ax.set_title(f"EmPath Stacked Fusion — Confusion Matrix\n"
             f"LOSO-67 Reactive Subjects  |  Acc = {overall_acc*100:.1f}%",
             fontsize=11, pad=12)
ax.set_ylabel("True Label", fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
plt.tight_layout()
out_cm = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(out_cm, dpi=150)
plt.close()
print(f"  Saved → {out_cm}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOT 2 — Per-Subject Accuracy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

print("Generating per-subject accuracy chart...")
subj_df = (pd.DataFrame([{"subject_id": k, "accuracy": v}
                          for k, v in subject_accs.items()])
           .sort_values("accuracy", ascending=True)
           .reset_index(drop=True))

colors = ["#d32f2f" if a < 0.5
          else "#f57c00" if a < 0.65
          else "#388e3c"
          for a in subj_df["accuracy"]]

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(subj_df)), subj_df["accuracy"], color=colors, width=0.8)
ax.axhline(0.5,         color="#d32f2f", linestyle="--", lw=1.5, alpha=0.8,
           label="Chance (50%)")
ax.axhline(overall_acc, color="#1976d2", linestyle="--", lw=1.5, alpha=0.8,
           label=f"Mean ({overall_acc*100:.1f}%)")
ax.set_xlim(-1, len(subj_df))
ax.set_ylim(0, 1.05)
ax.set_xlabel("Subject (sorted by accuracy)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Per-Subject Accuracy — EmPath Stacked Fusion\n"
             "Red < 50% | Orange 50–65% | Green ≥ 65%", fontsize=11)
ax.legend(fontsize=10)

n_above_80  = sum(1 for a in subj_df["accuracy"] if a >= 0.8)
n_above_65  = sum(1 for a in subj_df["accuracy"] if 0.65 <= a < 0.8)
n_chance    = sum(1 for a in subj_df["accuracy"] if 0.5 <= a < 0.65)
n_below     = sum(1 for a in subj_df["accuracy"] if a < 0.5)
ax.text(0.01, 0.97,
        f"≥80%: {n_above_80} | 65–80%: {n_above_65} | 50–65%: {n_chance} | <50%: {n_below}",
        transform=ax.transAxes, va="top", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
plt.tight_layout()
out_subj = os.path.join(OUTPUT_DIR, "per_subject_accuracy.png")
plt.savefig(out_subj, dpi=150)
plt.close()
print(f"  Saved → {out_subj}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOT 3 — SHAP Summary (bar) — Biosignal RF
# ══════════════════════════════════════════════════════════════════════════════

print("Generating SHAP biosignal summary (bar)...")
mean_shap_bio = np.abs(shap_bio_all).mean(axis=0)
shap_bio_df   = pd.DataFrame({"feature": bio_cols, "mean_shap": mean_shap_bio})
shap_bio_df   = shap_bio_df.sort_values("mean_shap", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette("Blues_r", len(shap_bio_df))
ax.barh(range(len(shap_bio_df)), shap_bio_df["mean_shap"].values[::-1],
        color=palette[::-1])
ax.set_yticks(range(len(shap_bio_df)))
ax.set_yticklabels(shap_bio_df["feature"].values[::-1], fontsize=10)
ax.set_xlabel("Mean |SHAP value|  (impact on PA3 prediction)", fontsize=11)
ax.set_title("SHAP Feature Importance — Biosignal RF\n"
             "Top 20 features driving PA2 vs PA3 classification", fontsize=11)
ax.axvline(0, color="black", lw=0.5)
plt.tight_layout()
out_shap_bio_bar = os.path.join(OUTPUT_DIR, "shap_biosignal_bar.png")
plt.savefig(out_shap_bio_bar, dpi=150)
plt.close()
print(f"  Saved → {out_shap_bio_bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOT 4 — SHAP Beeswarm — Biosignal RF (top 15 features)
# ══════════════════════════════════════════════════════════════════════════════

print("Generating SHAP beeswarm — biosignal RF...")
top_bio_idx  = np.argsort(mean_shap_bio)[::-1][:15]
top_bio_names = [bio_cols[i] for i in top_bio_idx]

fig, ax = plt.subplots(figsize=(10, 7))
# Manual beeswarm-style dot plot (SHAP value vs feature, colored by feature value)
for rank, (fi, fname) in enumerate(zip(top_bio_idx, top_bio_names)):
    sv   = shap_bio_all[:, fi]          # SHAP values for this feature
    fv   = X_bio_all[:, fi]             # normalized feature values
    # Jitter y
    jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
    # Normalize feature values for color mapping
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
    sc = ax.scatter(sv, rank + jitter, c=fv_norm, cmap="RdBu_r",
                    alpha=0.4, s=8, vmin=0, vmax=1)

ax.set_yticks(range(15))
ax.set_yticklabels(top_bio_names, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("SHAP value  (positive = pushes toward PA3)", fontsize=10)
ax.set_title("SHAP Beeswarm — Biosignal RF (Top 15 Features)\n"
             "Color: Blue = low feature value, Red = high feature value", fontsize=10)
cbar = plt.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label("Feature value (normalized)", fontsize=9)
plt.tight_layout()
out_shap_bio_bees = os.path.join(OUTPUT_DIR, "shap_biosignal_beeswarm.png")
plt.savefig(out_shap_bio_bees, dpi=150)
plt.close()
print(f"  Saved → {out_shap_bio_bees}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOT 5 — SHAP Summary (bar) — Landmark RF
# ══════════════════════════════════════════════════════════════════════════════

print("Generating SHAP landmark summary (bar)...")
mean_shap_lm = np.abs(shap_lm_all).mean(axis=0)
shap_lm_df   = pd.DataFrame({"feature": lm_cols, "mean_shap": mean_shap_lm})
shap_lm_df   = shap_lm_df.sort_values("mean_shap", ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette("Greens_r", len(shap_lm_df))
ax.barh(range(len(shap_lm_df)), shap_lm_df["mean_shap"].values[::-1],
        color=palette[::-1])
ax.set_yticks(range(len(shap_lm_df)))
ax.set_yticklabels(shap_lm_df["feature"].values[::-1], fontsize=9)
ax.set_xlabel("Mean |SHAP value|  (impact on PA3 prediction)", fontsize=11)
ax.set_title("SHAP Feature Importance — Landmark RF\n"
             "Facial geometry features driving PA2 vs PA3 classification", fontsize=11)
ax.axvline(0, color="black", lw=0.5)
plt.tight_layout()
out_shap_lm_bar = os.path.join(OUTPUT_DIR, "shap_landmark_bar.png")
plt.savefig(out_shap_lm_bar, dpi=150)
plt.close()
print(f"  Saved → {out_shap_lm_bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PLOT 6 — SHAP Beeswarm — Landmark RF
# ══════════════════════════════════════════════════════════════════════════════

print("Generating SHAP beeswarm — landmark RF...")
top_lm_n     = min(15, len(lm_cols))
top_lm_idx   = np.argsort(mean_shap_lm)[::-1][:top_lm_n]
top_lm_names = [lm_cols[i] for i in top_lm_idx]

fig, ax = plt.subplots(figsize=(10, 7))
for rank, (fi, fname) in enumerate(zip(top_lm_idx, top_lm_names)):
    sv      = shap_lm_all[:, fi]
    fv      = X_lm_all[:, fi]
    jitter  = np.random.uniform(-0.3, 0.3, size=len(sv))
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
    sc = ax.scatter(sv, rank + jitter, c=fv_norm, cmap="RdBu_r",
                    alpha=0.4, s=8, vmin=0, vmax=1)

ax.set_yticks(range(top_lm_n))
ax.set_yticklabels(top_lm_names, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("SHAP value  (positive = pushes toward PA3)", fontsize=10)
ax.set_title("SHAP Beeswarm — Landmark RF (Top Facial Features)\n"
             "Color: Blue = low feature value, Red = high feature value", fontsize=10)
cbar = plt.colorbar(sc, ax=ax, pad=0.01)
cbar.set_label("Feature value (normalized)", fontsize=9)
plt.tight_layout()
out_shap_lm_bees = os.path.join(OUTPUT_DIR, "shap_landmark_beeswarm.png")
plt.savefig(out_shap_lm_bees, dpi=150)
plt.close()
print(f"  Saved → {out_shap_lm_bees}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. PLOT 7 — Combined RF Feature Importance (RF importance, not SHAP)
# ══════════════════════════════════════════════════════════════════════════════

print("Generating combined RF feature importance...")
# Re-run a single full-data RF (not LOSO) for a quick importance reference
X_combined = np.hstack([X_bio, X_lm])
all_feat_names = bio_cols + lm_cols

scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

rf_full = RandomForestClassifier(
    n_estimators=300, max_depth=4, min_samples_split=10,
    max_features="sqrt", random_state=SEED, n_jobs=-1
)
rf_full.fit(X_combined_scaled, y)
imp_df = (pd.DataFrame({"feature": all_feat_names,
                         "importance": rf_full.feature_importances_})
          .sort_values("importance", ascending=False)
          .head(20))

fig, ax = plt.subplots(figsize=(12, 7))
colors_imp = ["#1565c0" if f in bio_cols else "#2e7d32"
              for f in imp_df["feature"]]
ax.bar(range(len(imp_df)), imp_df["importance"], color=colors_imp)
ax.set_xticks(range(len(imp_df)))
ax.set_xticklabels(imp_df["feature"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("RF Gini Importance", fontsize=11)
ax.set_title("Top 20 Feature Importances — Combined RF\nBlue = Biosignal | Green = Landmark",
             fontsize=11)
ax.legend(handles=[mpatches.Patch(color="#1565c0", label="Biosignal"),
                   mpatches.Patch(color="#2e7d32", label="Landmark")],
          fontsize=10)
plt.tight_layout()
out_imp = os.path.join(OUTPUT_DIR, "feature_importance_combined.png")
plt.savefig(out_imp, dpi=150)
plt.close()
print(f"  Saved → {out_imp}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. SAVE CSVs
# ══════════════════════════════════════════════════════════════════════════════

subj_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_accuracy.csv"), index=False)

shap_bio_df.to_csv(os.path.join(OUTPUT_DIR, "shap_biosignal_ranked.csv"), index=False)
shap_lm_df.to_csv(os.path.join(OUTPUT_DIR, "shap_landmark_ranked.csv"),  index=False)

# Full SHAP matrix
pd.DataFrame(shap_bio_all, columns=bio_cols).to_csv(
    os.path.join(OUTPUT_DIR, "shap_biosignal_matrix.csv"), index=False)
pd.DataFrame(shap_lm_all, columns=lm_cols).to_csv(
    os.path.join(OUTPUT_DIR, "shap_landmark_matrix.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 13. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("  SHAP ANALYSIS COMPLETE")
print(f"{'=' * 65}")
print(f"  LOSO accuracy       : {overall_acc*100:.1f}%")
print(f"  Best subjects (≥80%): {n_above_80}")
print(f"  Good  subjects (65–80%): {n_above_65}")
print(f"  Chance (50–65%)     : {n_chance}")
print(f"  Below chance (<50%) : {n_below}")
print(f"\n  Top 5 biosignal features by |SHAP|:")
for _, row in shap_bio_df.head(5).iterrows():
    print(f"    {row['feature']:30s}  {row['mean_shap']:.4f}")
print(f"\n  Top 5 landmark features by |SHAP|:")
for _, row in shap_lm_df.head(5).iterrows():
    print(f"    {row['feature']:30s}  {row['mean_shap']:.4f}")
print(f"\n  Output files → {OUTPUT_DIR}/")
print("    confusion_matrix.png")
print("    per_subject_accuracy.png")
print("    shap_biosignal_bar.png")
print("    shap_biosignal_beeswarm.png")
print("    shap_landmark_bar.png")
print("    shap_landmark_beeswarm.png")
print("    feature_importance_combined.png")
print("    per_subject_accuracy.csv")
print("    shap_biosignal_ranked.csv")
print("    shap_landmark_ranked.csv")
print(f"{'=' * 65}")
print("Done.")
