"""
Landmark Velocity Features — Optical Flow Proxy on MediaPipe Landmarks
======================================================================
Novel contribution: instead of static landmark positions (where the face is),
use VELOCITY of landmarks (how fast the face is moving) to detect PA2 vs PA3.

WHY THIS MATTERS FOR PA2 vs PA3:
  PA2 and PA3 differ by only ~1°C thermal stimulus.
  Static facial geometry may look nearly identical between the two levels.
  But the DYNAMICS differ — PA3 may produce faster brow furrowing, quicker
  mouth tightening, more rapid cheek contraction.

  This is the optical flow insight from Thiam et al. (2020): motion history
  carries pain signal that static frames miss.

IMPLEMENTATION:
  We already have raw_coords.npz: (N, 24, 468, 2) — 24 frames per sample.
  Velocity = frame[t+1] - frame[t] → (N, 23, 468, 2) displacement vectors.

  No raw video reprocessing needed — derived entirely from existing coords.

FEATURES EXTRACTED (72 per sample):
  Per-frame motion statistics (across all 468 landmarks):
    - mean magnitude  (N, 23): average landmark movement per frame
    - std magnitude   (N, 23): variation in landmark movement per frame
    - max magnitude   (N, 23): peak landmark movement per frame
  Global statistics (across all frames + landmarks):
    - global mean, std, max: (N, 3)
  Total: 23×3 + 3 = 72 velocity features

EVALUATION:
  1. RF on velocity features only  (compare to RF landmarks: 61.4%)
  2. RF velocity + RF biosignal stacked  (compare to baseline: 65.3%)

Usage:
    python SRC/preprocessing/evaluate_velocity_loso.py

Runtime: ~2 min (feature extraction + 67-fold RF LOSO, no GPU needed).
"""

import os, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COORDS_NPZ  = os.path.join(BASE_DIR, "Results", "landmarks_gnn",   "raw_coords.npz")
BIOSIG_CSV  = os.path.join(BASE_DIR, "Results", "biosignals_hrv",  "all_67_hrv.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "velocity_loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD COORDS + COMPUTE VELOCITY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("Loading raw coordinate data...")
if not os.path.exists(COORDS_NPZ):
    print(f"ERROR: {COORDS_NPZ} not found.")
    print("Run extract_landmarks_raw_coords.py first.")
    sys.exit(1)

npz        = np.load(COORDS_NPZ, allow_pickle=True)
coords     = npz["coords"].astype(np.float32)   # (N, 24, 468, 2)
labels_all = npz["labels"].astype(np.int64)
groups_all = npz["subject_ids"].astype(np.int64)
snames_all = npz["sample_names"]
print(f"  Loaded: {len(labels_all)} samples, {np.unique(groups_all).size} subjects")
print(f"  Coords shape: {coords.shape}")

# ── Compute velocity (frame differences) ──────────────────────────────────────
print("\nComputing landmark velocity features...")
# velocity[t] = position[t+1] - position[t]
velocity = coords[:, 1:, :, :] - coords[:, :-1, :, :]   # (N, 23, 468, 2)

# Euclidean magnitude of displacement per landmark per frame
mag = np.linalg.norm(velocity, axis=-1)                   # (N, 23, 468)

# Per-frame statistics (collapses 468 landmarks → scalar per frame)
frame_mean = mag.mean(axis=2)   # (N, 23) — avg motion per frame
frame_std  = mag.std(axis=2)    # (N, 23) — variation in motion per frame
frame_max  = mag.max(axis=2)    # (N, 23) — peak motion per frame

# Global statistics (collapses all frames + landmarks → 3 scalars)
global_mean = mag.mean(axis=(1, 2)).reshape(-1, 1)  # (N, 1)
global_std  = mag.std(axis=(1, 2)).reshape(-1, 1)   # (N, 1)
global_max  = mag.max(axis=(1, 2)).reshape(-1, 1)   # (N, 1)

# Concatenate all velocity features
X_vel = np.hstack([
    frame_mean,   # (N, 23)
    frame_std,    # (N, 23)
    frame_max,    # (N, 23)
    global_mean,  # (N, 1)
    global_std,   # (N, 1)
    global_max,   # (N, 1)
])   # (N, 72)

print(f"  Velocity features shape : {X_vel.shape}")
print(f"  Magnitude range         : [{mag.min():.4f}, {mag.max():.4f}]")
print(f"  Mean per-frame motion   : {frame_mean.mean():.4f} ± {frame_mean.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALIGN WITH BIOSIGNAL DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading biosignal features...")
bio_df   = pd.read_csv(BIOSIG_CSV)
EXCLUDE  = {"subject_id", "sample_name", "class_name", "label"}
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]
bio_lu   = {row["sample_name"]: row[bio_cols].values.astype(np.float32)
            for _, row in bio_df.iterrows()}

# Keep only samples present in both coords and biosignal
valid    = np.array([s in bio_lu for s in snames_all])
X_vel    = X_vel[valid]
labels_all = labels_all[valid]
groups_all = groups_all[valid]
snames_all = snames_all[valid]
X_bio    = np.nan_to_num(
    np.array([bio_lu[s] for s in snames_all], dtype=np.float32)
)
print(f"  Matched: {len(labels_all)} samples, {X_bio.shape[1]} biosignal features")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def person_norm_train(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = X.copy()
    for sid in np.unique(groups):
        mask      = groups == sid
        mean, std = X[mask].mean(0), X[mask].std(0)
        std[std == 0] = 1.0
        out[mask] = (X[mask] - mean) / std
    return out

def person_norm_test(X: np.ndarray) -> np.ndarray:
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    return (X - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

logo    = LeaveOneGroupOut()
n_folds = np.unique(groups_all).size
print(f"\nRunning LOSO ({n_folds} folds)...")
print("=" * 60)

vel_accs     = []   # RF on velocity features only
stacked_accs = []   # RF_vel + RF_bio → LogReg stacked

fold = 0
for train_idx, test_idx in logo.split(X_vel, labels_all, groups_all):
    fold += 1
    g_train = groups_all[train_idx]
    y_train = labels_all[train_idx]
    y_test  = labels_all[test_idx]

    # ── Normalise ─────────────────────────────────────────────────────────
    X_vel_train  = person_norm_train(X_vel[train_idx],  g_train)
    X_vel_test   = person_norm_test(X_vel[test_idx])
    X_bio_train  = person_norm_train(X_bio[train_idx],  g_train)
    X_bio_test   = person_norm_test(X_bio[test_idx])

    # ── RF on velocity features ───────────────────────────────────────────
    rf_vel = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf_vel.fit(X_vel_train, y_train)
    vel_acc = accuracy_score(y_test, rf_vel.predict(X_vel_test))
    vel_accs.append(vel_acc)

    # ── RF on biosignals ──────────────────────────────────────────────────
    rf_bio = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf_bio.fit(X_bio_train, y_train)

    # ── Stacked: RF_vel probs + RF_bio probs → LogReg ─────────────────────
    vel_train_probs = rf_vel.predict_proba(X_vel_train)
    bio_train_probs = rf_bio.predict_proba(X_bio_train)
    vel_test_probs  = rf_vel.predict_proba(X_vel_test)
    bio_test_probs  = rf_bio.predict_proba(X_bio_test)

    X_meta_train = np.hstack([vel_train_probs, bio_train_probs])
    X_meta_test  = np.hstack([vel_test_probs,  bio_test_probs])
    meta = LogisticRegression(max_iter=1000, random_state=SEED)
    meta.fit(X_meta_train, y_train)
    stacked_acc = accuracy_score(y_test, meta.predict(X_meta_test))
    stacked_accs.append(stacked_acc)

    print(f"  Fold {fold:2d}/{n_folds} | "
          f"Velocity RF: {vel_acc*100:.1f}% | "
          f"Stacked (vel+bio): {stacked_acc*100:.1f}% | "
          f"Running stacked mean: {np.mean(stacked_accs)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESULTS
# ══════════════════════════════════════════════════════════════════════════════

vel_mean,     vel_std     = np.mean(vel_accs)     * 100, np.std(vel_accs)     * 100
stacked_mean, stacked_std = np.mean(stacked_accs) * 100, np.std(stacked_accs) * 100

print(f"\n{'=' * 60}")
print(f"  LANDMARK VELOCITY LOSO RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline: RF landmarks (static flat)    61.4% ± 13.1%")
print(f"  Baseline: RF biosignal only             63.1% ± 11.6%")
print(f"  Baseline: Stacked fusion (RF+RF+LR)     65.3% ± 14.1%")
print(f"{'=' * 60}")
print(f"  Velocity RF only           : {vel_mean:.1f}% ± {vel_std:.1f}%")
print(f"  Velocity RF + Biosig RF    : {stacked_mean:.1f}% ± {stacked_std:.1f}%")
print(f"{'=' * 60}")
print(f"  Δ Velocity vs static lm    : {vel_mean - 61.4:+.1f}%")
print(f"  Δ Stacked vs baseline      : {stacked_mean - 65.3:+.1f}%")
print(f"  Δ Std vs baseline          : {stacked_std - 14.1:+.1f}%")
print(f"{'=' * 60}")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({
    "fold":        range(1, n_folds + 1),
    "vel_acc":     vel_accs,
    "stacked_acc": stacked_accs,
}).to_csv(os.path.join(RESULTS_DIR, "velocity_loso_results.csv"), index=False)

pd.DataFrame([{
    "vel_mean": vel_mean, "vel_std": vel_std,
    "stacked_mean": stacked_mean, "stacked_std": stacked_std,
}]).to_csv(os.path.join(RESULTS_DIR, "velocity_loso_summary.csv"), index=False)

print(f"\nResults saved → {RESULTS_DIR}")
print("Done.")
