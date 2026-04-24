"""
CrossMod — Cross-Modal Attention Fusion (Biosignals ↔ Landmarks)
================================================================
Novel contribution: proper cross-modal attention between biosignal and
landmark embeddings, replacing the LogReg meta-learner in the stacked fusion.

WHY THE PREVIOUS ATTENTION FUSION FAILED (61.1%):
  The previous implementation used self-attention within each modality —
  biosignals attending to biosignals, landmarks to landmarks. That's just
  a weighted combination of the same features (barely better than MLP).

WHAT CROSSMOD DOES DIFFERENTLY:
  Bidirectional cross-modal attention:
    - Biosignal embedding queries landmark embedding (K, V)
      → "which landmark patterns confirm what the biosignal is saying?"
    - Landmark embedding queries biosignal embedding (K, V)
      → "which physiological patterns confirm what the face is showing?"

  The attended representations are fused and fed to the pain classifier.
  The model learns to align both modalities in a shared pain-relevant space.

Architecture:
    Input          : 35 biosignal features + 22 landmark features
    bio_encoder    : FC(35→64)+BN+ReLU+Drop → FC(64→64)+BN+ReLU
    lm_encoder     : FC(22→64)+BN+ReLU+Drop → FC(64→64)+BN+ReLU

    cross_attn_b2l : MultiheadAttention(embed=64, heads=4) [bio queries lm]
    cross_attn_l2b : MultiheadAttention(embed=64, heads=4) [lm queries bio]

    Residual+Norm  : LayerNorm(bio_emb + bio_attended)
                   LayerNorm(lm_emb  + lm_attended)

    classifier     : FC(128→32)+ReLU+Drop(0.3) → FC(32→2)

Evaluation:
    CrossMod end-to-end (compare to stacked fusion baseline 65.3% ± 14.1%)

Usage:
    python SRC/preprocessing/evaluate_crossmod_loso.py

Runtime: ~5–15 min on GPU (tiny model, 67 folds × 100 epochs).
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
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIOSIG_CSV  = os.path.join(BASE_DIR, "Results", "biosignals_hrv",   "all_67_hrv.csv")
LM_CSV      = os.path.join(BASE_DIR, "Results", "landmarks_all67",  "landmarks_all67.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "crossmod_loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 100   # cross-attention needs more epochs than RF to converge
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.3
EMB_DIM     = 64    # shared embedding dim for both modalities
NUM_HEADS   = 4     # attention heads (must divide EMB_DIM)
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CrossModFusion(nn.Module):
    """
    Bidirectional cross-modal attention fusion for pain classification.

    Each modality is encoded to a shared EMB_DIM space, then:
      bio_emb   queries lm_emb  (bio-to-landmark attention)
      lm_emb    queries bio_emb (landmark-to-bio attention)
    Attended outputs are residual-summed, layer-normed, concatenated,
    then classified.
    """

    def __init__(self, bio_in: int, lm_in: int,
                 emb_dim: int = EMB_DIM,
                 num_heads: int = NUM_HEADS,
                 dropout: float = DROPOUT):
        super().__init__()

        # ── Modality encoders ──────────────────────────────────────────────
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_in, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )
        self.lm_encoder = nn.Sequential(
            nn.Linear(lm_in, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        # ── Cross-modal attention heads ────────────────────────────────────
        # bio-to-landmark: biosignal embedding queries landmark embedding
        self.attn_b2l = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        # landmark-to-bio: landmark embedding queries biosignal embedding
        self.attn_l2b = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # ── Post-attention normalisation (Pre-LN style) ────────────────────
        self.norm_bio = nn.LayerNorm(emb_dim)
        self.norm_lm  = nn.LayerNorm(emb_dim)

        # ── Pain classifier on fused representation ────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x_bio: torch.Tensor, x_lm: torch.Tensor):
        """
        Args:
            x_bio : (B, bio_in)   biosignal features
            x_lm  : (B, lm_in)    landmark features

        Returns:
            logits : (B, 2)
        """
        bio_emb = self.bio_encoder(x_bio)   # (B, emb_dim)
        lm_emb  = self.lm_encoder(x_lm)    # (B, emb_dim)

        # MultiheadAttention expects (B, seq_len, emb_dim)
        # For feature vectors, seq_len = 1
        bio_seq = bio_emb.unsqueeze(1)      # (B, 1, emb_dim)
        lm_seq  = lm_emb.unsqueeze(1)       # (B, 1, emb_dim)

        # Biosignal queries landmarks
        bio_ctx, _ = self.attn_b2l(query=bio_seq, key=lm_seq, value=lm_seq)
        bio_ctx    = bio_ctx.squeeze(1)     # (B, emb_dim)

        # Landmark queries biosignals
        lm_ctx, _  = self.attn_l2b(query=lm_seq,  key=bio_seq, value=bio_seq)
        lm_ctx     = lm_ctx.squeeze(1)     # (B, emb_dim)

        # Residual + LayerNorm
        bio_out = self.norm_bio(bio_emb + bio_ctx)   # (B, emb_dim)
        lm_out  = self.norm_lm(lm_emb  + lm_ctx)    # (B, emb_dim)

        # Concatenate + classify
        fused  = torch.cat([bio_out, lm_out], dim=-1)   # (B, 2*emb_dim)
        return self.classifier(fused)                    # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALISATION
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
# 3. TRAINING + INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def train_crossmod(X_bio: np.ndarray, X_lm: np.ndarray,
                   y: np.ndarray, device: torch.device) -> CrossModFusion:
    model     = CrossModFusion(bio_in=X_bio.shape[1], lm_in=X_lm.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    N       = len(y)
    indices = np.arange(N)

    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(indices)

        for start in range(0, N, BATCH_SIZE):
            idx    = indices[start: start + BATCH_SIZE]
            xb_t   = torch.tensor(X_bio[idx], dtype=torch.float32).to(device)
            xl_t   = torch.tensor(X_lm[idx],  dtype=torch.float32).to(device)
            y_t    = torch.tensor(y[idx],      dtype=torch.long).to(device)

            logits = model(xb_t, xl_t)
            loss   = F.cross_entropy(logits, y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    return model


@torch.no_grad()
def predict(model: CrossModFusion,
            X_bio: np.ndarray, X_lm: np.ndarray,
            device: torch.device) -> np.ndarray:
    model.eval()
    all_preds = []
    for start in range(0, len(X_bio), BATCH_SIZE):
        xb_t = torch.tensor(X_bio[start: start + BATCH_SIZE], dtype=torch.float32).to(device)
        xl_t = torch.tensor(X_lm[start:  start + BATCH_SIZE], dtype=torch.float32).to(device)
        preds = model(xb_t, xl_t).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
    return np.concatenate(all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading biosignal features...")
bio_df   = pd.read_csv(BIOSIG_CSV)
EXCLUDE  = {"subject_id", "sample_name", "class_name", "label"}
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]
X_bio_raw = np.nan_to_num(bio_df[bio_cols].values.astype(np.float32))
y_all     = bio_df["label"].values.astype(np.int64)
groups    = bio_df["subject_id"].values.astype(np.int64)
snames    = bio_df["sample_name"].values
print(f"  {len(y_all)} samples, {np.unique(groups).size} subjects, {X_bio_raw.shape[1]} features")

print("Loading flat landmark features...")
if not os.path.exists(LM_CSV):
    print(f"ERROR: {LM_CSV} not found.")
    sys.exit(1)
lm_df   = pd.read_csv(LM_CSV)
lm_cols = [c for c in lm_df.columns if c not in EXCLUDE]
lm_lu   = {row["sample_name"]: row[lm_cols].values.astype(np.float32)
           for _, row in lm_df.iterrows()}

# Align both modalities by sample_name
valid     = np.array([s in lm_lu for s in snames])
X_bio_raw = X_bio_raw[valid]
y_all     = y_all[valid]
groups    = groups[valid]
snames    = snames[valid]
X_lm_raw  = np.nan_to_num(
    np.array([lm_lu[s] for s in snames], dtype=np.float32)
)
print(f"  Matched: {len(snames)} samples, {X_lm_raw.shape[1]} landmark features")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

logo    = LeaveOneGroupOut()
n_folds = np.unique(groups).size
print(f"\nRunning LOSO ({n_folds} folds) — CrossMod cross-attention fusion...")
print("=" * 60)

accs = []
fold = 0

for train_idx, test_idx in logo.split(X_bio_raw, y_all, groups):
    fold += 1
    g_train = groups[train_idx]
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    # ── Person-specific normalisation ─────────────────────────────────────
    X_bio_train = person_norm_train(X_bio_raw[train_idx], g_train)
    X_bio_test  = person_norm_test(X_bio_raw[test_idx])
    X_lm_train  = person_norm_train(X_lm_raw[train_idx],  g_train)
    X_lm_test   = person_norm_test(X_lm_raw[test_idx])

    # ── Train CrossMod ────────────────────────────────────────────────────
    model = train_crossmod(X_bio_train, X_lm_train, y_train, DEVICE)

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds = predict(model, X_bio_test, X_lm_test, DEVICE)
    acc   = accuracy_score(y_test, preds)
    accs.append(acc)

    print(f"  Fold {fold:2d}/{n_folds} | CrossMod: {acc*100:.1f}% | "
          f"Running mean: {np.mean(accs)*100:.1f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS
# ══════════════════════════════════════════════════════════════════════════════

mean_acc = np.mean(accs) * 100
std_acc  = np.std(accs)  * 100

print(f"\n{'=' * 60}")
print(f"  CROSSMOD LOSO RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline: RF biosignal only          63.1% ± 11.6%")
print(f"  Baseline: Stacked fusion (RF+RF+LR)  65.3% ± 14.1%")
print(f"{'=' * 60}")
print(f"  CrossMod cross-attention : {mean_acc:.1f}% ± {std_acc:.1f}%")
print(f"  Δ vs stacked baseline    : {mean_acc - 65.3:+.1f}%")
print(f"  Δ std vs baseline        : {std_acc - 14.1:+.1f}%")
print(f"{'=' * 60}")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame({"fold": range(1, n_folds + 1), "acc": accs}).to_csv(
    os.path.join(RESULTS_DIR, "crossmod_loso_results.csv"), index=False
)
pd.DataFrame([{"mean_acc": mean_acc, "std_acc": std_acc}]).to_csv(
    os.path.join(RESULTS_DIR, "crossmod_loso_summary.csv"), index=False
)
print(f"\nResults saved → {RESULTS_DIR}")
print("Done.")
