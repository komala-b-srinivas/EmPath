"""
DANN — Domain-Adversarial Neural Network on Biosignal Features
==============================================================
Novel contribution: adversarial domain adaptation for subject-invariant
pain feature learning.

Problem: Each person has a different physiological baseline (GSR, ECG, EMG).
Standard models memorise subject identity → high per-fold variance (±14.1%).

Solution (Ganin et al., 2016 — DANN):
  Shared Encoder F  →  Pain Classifier C_y   (minimise pain CE loss)
                    →  Gradient Reversal Layer
                    →  Subject Classifier C_d  (maximise subject confusion)

The GRL flips the sign of gradients flowing back through it, forcing F to
learn features that are simultaneously:
  (a) predictive of pain  (C_y maximises accuracy)
  (b) useless for identifying the subject  (GRL + C_d adversarial pressure)

Architecture:
    Input     : 35 biosignal features (from all_67_hrv.csv)
    Encoder F : FC(35→64)+BN+ReLU+Drop → FC(64→32)+BN+ReLU
    Pain C_y  : FC(32→2)
    Subject   : GRL → FC(32→32)+ReLU → FC(32→n_train_subjects)

LOSO evaluation:
    For each fold, 66 subjects train the full DANN.
    Subject labels are re-indexed 0..65 for each fold.
    At test time the GRL is off — only the encoder + pain head run.
    We also stack DANN pain probs with flat-landmark RF probs and compare
    to the baseline stacked fusion (65.3% ± 14.1%).

Expected outcomes vs baseline:
    - Accuracy : hopefully +2–5% from subject-invariant features
    - Std      : hopefully ±10–11% from reduced domain shift (key metric)

Usage:
    python SRC/preprocessing/evaluate_dann_loso.py

Runtime: ~10–20 min on GPU (LOSO = 67 folds × 100 epochs).
"""

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIOSIG_CSV  = os.path.join(BASE_DIR, "Results", "biosignals_hrv", "all_67_hrv.csv")
LM_CSV      = os.path.join(BASE_DIR, "Results", "landmarks_all67", "landmarks_all67.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "dann_loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS       = 100   # DANN needs more epochs than standard; GRL schedule goes 0→1
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
ENC_H1       = 64    # encoder layer 1 width
ENC_H2       = 32    # encoder output (shared representation) width
ALPHA_MAX    = 1.0   # GRL scale at end of training (Ganin et al. recommend 1.0)
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRADIENT REVERSAL LAYER
# ══════════════════════════════════════════════════════════════════════════════

class _GRLFunction(torch.autograd.Function):
    """
    Forward  : identity  (passes input through unchanged)
    Backward : negates + scales gradient by alpha

    This is the key mechanism: the encoder sees a gradient that tells it to
    make subject features LESS discriminative, while the subject classifier
    sees the normal gradient (trying to classify subjects).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None   # flip + scale


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return _GRLFunction.apply(x, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DANN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network for pain classification.

    encoder:          35 → 64 → 32  (shared subject-invariant features)
    pain_classifier:  32 → 2        (PA2 vs PA3)
    subject_classifier: 32 → n_subjects  (adversarial; driven through GRL)
    """

    def __init__(self, in_features: int, n_subjects: int,
                 h1: int = ENC_H1, h2: int = ENC_H2, dropout: float = DROPOUT):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
        )

        self.pain_classifier = nn.Linear(h2, 2)

        self.grl = GradientReversalLayer()
        self.subject_classifier = nn.Sequential(
            nn.Linear(h2, h2),
            nn.ReLU(),
            nn.Linear(h2, n_subjects),
        )

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        """
        Args:
            x     : (B, in_features) biosignal feature vector
            alpha : GRL scale — 0 at training start, increases to ALPHA_MAX

        Returns:
            pain_logits    : (B, 2)
            subject_logits : (B, n_subjects)
        """
        feat            = self.encoder(x)
        pain_logits     = self.pain_classifier(feat)
        feat_reversed   = self.grl(feat, alpha)
        subject_logits  = self.subject_classifier(feat_reversed)
        return pain_logits, subject_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only — no GRL needed."""
        feat = self.encoder(x)
        return F.softmax(self.pain_classifier(feat), dim=1)   # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALPHA SCHEDULE (Ganin et al., 2016)
# ══════════════════════════════════════════════════════════════════════════════

def get_alpha(epoch: int, total_epochs: int, alpha_max: float = ALPHA_MAX) -> float:
    """
    Linearly ramps alpha from 0 to alpha_max over training.
    Starting with alpha=0 lets the pain classifier stabilise before
    adversarial pressure kicks in.
    """
    p = epoch / max(total_epochs - 1, 1)
    return alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. NORMALISATION (person-specific, same strategy as baseline)
# ══════════════════════════════════════════════════════════════════════════════

def person_norm_train(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = X.copy()
    for sid in np.unique(groups):
        mask      = groups == sid
        mean, std = X[mask].mean(axis=0), X[mask].std(axis=0)
        std[std == 0] = 1.0
        out[mask] = (X[mask] - mean) / std
    return out


def person_norm_test(X: np.ndarray) -> np.ndarray:
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING + INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def train_dann(X_train: np.ndarray, y_pain: np.ndarray,
               subject_ids: np.ndarray, device: torch.device) -> DANN:
    """
    Train one DANN fold.

    subject_ids : (N,) — original IDs; re-indexed inside to 0..n_subjects-1
    Returns trained DANN model.
    """
    # Re-index subjects to 0..K-1 for this fold's classifier
    unique_subs  = np.unique(subject_ids)
    sub_remap    = {s: i for i, s in enumerate(unique_subs)}
    y_subject    = np.array([sub_remap[s] for s in subject_ids], dtype=np.int64)
    n_subjects   = len(unique_subs)

    model     = DANN(in_features=X_train.shape[1], n_subjects=n_subjects).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    N       = len(X_train)
    indices = np.arange(N)

    for epoch in range(EPOCHS):
        model.train()
        alpha = get_alpha(epoch, EPOCHS)
        np.random.shuffle(indices)

        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start: start + BATCH_SIZE]

            x_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(device)
            y_p     = torch.tensor(y_pain[batch_idx],    dtype=torch.long).to(device)
            y_d     = torch.tensor(y_subject[batch_idx], dtype=torch.long).to(device)

            pain_logits, subj_logits = model(x_batch, alpha=alpha)

            pain_loss    = F.cross_entropy(pain_logits, y_p)
            subject_loss = F.cross_entropy(subj_logits, y_d)
            loss         = pain_loss + subject_loss   # GRL handles the sign flip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    return model


@torch.no_grad()
def get_dann_probs(model: DANN, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return softmax pain probs (N, 2) for a feature matrix."""
    model.eval()
    all_probs = []
    for start in range(0, len(X), BATCH_SIZE):
        x_b  = torch.tensor(X[start: start + BATCH_SIZE], dtype=torch.float32).to(device)
        prob = model.predict_proba(x_b).cpu().numpy()
        all_probs.append(prob)
    return np.vstack(all_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading biosignal features...")
if not os.path.exists(BIOSIG_CSV):
    print(f"ERROR: {BIOSIG_CSV} not found.")
    sys.exit(1)

bio_df   = pd.read_csv(BIOSIG_CSV)
EXCLUDE  = {"subject_id", "sample_name", "class_name", "label"}
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]

X_bio    = np.nan_to_num(bio_df[bio_cols].values.astype(np.float32))
y_all    = bio_df["label"].values.astype(np.int64)
groups   = bio_df["subject_id"].values.astype(np.int64)
snames   = bio_df["sample_name"].values

print(f"  Samples  : {len(y_all)}")
print(f"  Subjects : {np.unique(groups).size}")
print(f"  Features : {X_bio.shape[1]}")

# ── Load flat landmark features for stacked fusion comparison ─────────────────
has_lm = os.path.exists(LM_CSV)
if has_lm:
    print("\nLoading flat landmark features for stacked fusion...")
    lm_df   = pd.read_csv(LM_CSV)
    lm_cols = [c for c in lm_df.columns if c not in EXCLUDE]
    lm_lu   = {row["sample_name"]: row[lm_cols].values.astype(np.float32)
               for _, row in lm_df.iterrows()}
    valid   = np.array([s in lm_lu for s in snames])
    if valid.sum() < len(snames):
        print(f"  Warning: {len(snames) - valid.sum()} biosignal samples have no landmark match — dropping")
        X_bio  = X_bio[valid]
        y_all  = y_all[valid]
        groups = groups[valid]
        snames = snames[valid]
    X_lm = np.nan_to_num(
        np.array([lm_lu[s] for s in snames], dtype=np.float32)
    )
    print(f"  Matched  : {len(snames)} samples, {X_lm.shape[1]} landmark features")
else:
    print("\nFlat landmark features not found — skipping stacked comparison")
    X_lm = None


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

logo    = LeaveOneGroupOut()
n_folds = np.unique(groups).size
print(f"\nRunning LOSO ({n_folds} folds) — DANN on biosignals...")
print("=" * 60)

dann_accs    = []   # DANN biosignal only
stacked_accs = []   # DANN bio probs + RF landmark probs → LogReg (if landmarks available)
rf_accs      = []   # RF biosignal only (sanity check — should match 63.1%)

fold = 0
for train_idx, test_idx in logo.split(X_bio, y_all, groups):
    fold += 1
    g_train = groups[train_idx]
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    # ── Person-specific normalisation ────────────────────────────────────────
    X_bio_train = person_norm_train(X_bio[train_idx], g_train)
    X_bio_test  = person_norm_test(X_bio[test_idx])

    # ── RF biosignal (sanity check baseline) ─────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf.fit(X_bio_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_bio_test))
    rf_accs.append(rf_acc)

    # ── Train DANN ────────────────────────────────────────────────────────────
    model = train_dann(X_bio_train, y_train, g_train, DEVICE)

    dann_test_probs  = get_dann_probs(model, X_bio_test, DEVICE)
    dann_train_probs = get_dann_probs(model, X_bio_train, DEVICE)

    dann_acc = accuracy_score(y_test, dann_test_probs.argmax(axis=1))
    dann_accs.append(dann_acc)

    # ── Stacked fusion: DANN probs + RF landmark probs → LogReg ──────────────
    if X_lm is not None:
        X_lm_train = person_norm_train(X_lm[train_idx], g_train)
        X_lm_test  = person_norm_test(X_lm[test_idx])

        rf_lm = RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_split=10,
            max_features="sqrt", random_state=SEED, n_jobs=-1
        )
        rf_lm.fit(X_lm_train, y_train)
        lm_train_probs = rf_lm.predict_proba(X_lm_train)
        lm_test_probs  = rf_lm.predict_proba(X_lm_test)

        # Meta-learner: [DANN_bio_probs | RF_lm_probs]
        X_meta_train = np.hstack([dann_train_probs, lm_train_probs])
        X_meta_test  = np.hstack([dann_test_probs,  lm_test_probs])
        meta = LogisticRegression(max_iter=1000, random_state=SEED)
        meta.fit(X_meta_train, y_train)
        stacked_acc = accuracy_score(y_test, meta.predict(X_meta_test))
        stacked_accs.append(stacked_acc)
        stacked_str = f" | Stacked: {stacked_acc*100:.1f}%"
    else:
        stacked_str = ""

    print(f"  Fold {fold:2d}/{n_folds} | RF: {rf_acc*100:.1f}% | "
          f"DANN: {dann_acc*100:.1f}%{stacked_str} | "
          f"Running DANN mean: {np.mean(dann_accs)*100:.1f}%")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESULTS
# ══════════════════════════════════════════════════════════════════════════════

rf_mean,   rf_std   = np.mean(rf_accs)   * 100, np.std(rf_accs)   * 100
dann_mean, dann_std = np.mean(dann_accs) * 100, np.std(dann_accs) * 100

print(f"\n{'=' * 60}")
print(f"  DANN LOSO RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline: RF biosignal (no adaptation)  63.1% ± 11.6%")
print(f"  Baseline: Stacked fusion (RF+RF+LR)     65.3% ± 14.1%")
print(f"{'=' * 60}")
print(f"  RF biosignal  (sanity check) : {rf_mean:.1f}% ± {rf_std:.1f}%")
print(f"  DANN biosignal               : {dann_mean:.1f}% ± {dann_std:.1f}%")
if stacked_accs:
    sm, ss = np.mean(stacked_accs) * 100, np.std(stacked_accs) * 100
    print(f"  DANN + RF landmarks stacked  : {sm:.1f}% ± {ss:.1f}%")
    print(f"{'=' * 60}")
    print(f"  Δ DANN vs RF biosignal    : {dann_mean - 63.1:+.1f}%")
    print(f"  Δ Stacked vs baseline     : {sm - 65.3:+.1f}%")
    print(f"  Δ Std vs baseline (key)   : {ss - 14.1:+.1f}%  (negative = less variance = better)")
print(f"{'=' * 60}")

# ── Save results ──────────────────────────────────────────────────────────────
fold_rows = []
for i, (da, ra) in enumerate(zip(dann_accs, rf_accs)):
    row = {"fold": i + 1, "dann_acc": da, "rf_acc": ra}
    if stacked_accs:
        row["stacked_acc"] = stacked_accs[i]
    fold_rows.append(row)

results_df = pd.DataFrame(fold_rows)
out_csv    = os.path.join(RESULTS_DIR, "dann_loso_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nPer-fold results saved: {out_csv}")

summary = {
    "rf_mean": rf_mean, "rf_std": rf_std,
    "dann_mean": dann_mean, "dann_std": dann_std,
}
if stacked_accs:
    summary["stacked_mean"] = sm
    summary["stacked_std"]  = ss
pd.DataFrame([summary]).to_csv(
    os.path.join(RESULTS_DIR, "dann_loso_summary.csv"), index=False
)
print(f"Summary saved        : {os.path.join(RESULTS_DIR, 'dann_loso_summary.csv')}")
print("\nDone.")
