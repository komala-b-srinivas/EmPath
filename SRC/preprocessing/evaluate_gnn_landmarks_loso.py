"""
GNN on Facial Landmarks — LOSO Evaluation  (EmPath v2, Week 3 experiment)
=========================================================================

Novel contribution: Instead of the 22 hand-engineered distance statistics used in
the baseline, we treat the 468 MediaPipe FaceMesh landmarks as nodes in an anatomical
graph and learn relational co-activation patterns via a Graph Attention Network (GAT).

The key insight: the baseline computes flat statistics like "average brow-eye distance."
A GNN can learn that "inner brow lowering COMBINED WITH mouth corner pulling" is a
pain pattern — relational information that flat statistics cannot capture.

Graph structure:
    Nodes  : 468 FaceMesh landmarks, features = (x, y) normalized coordinates
    Edges  : MediaPipe FACEMESH_TESSELATION — anatomical adjacency (~750 undirected edges)

Architecture:
    Per frame  : (468, 2) → GATConv → GATConv → GlobalMeanPool → (32,)
    Per sample : mean of 24 frame embeddings → (32,) sample embedding
    Classifier : Linear(32, 2) → PA2 / PA3

Evaluation:
    1. GNN landmarks only          (compare to RF landmarks: 61.4%)
    2. GNN landmarks + biosignal RF stacked  (compare to stacked fusion: 65.3%)

Requirements:
    pip install torch-geometric
    # On HPC with CUDA — check your torch + CUDA version first:
    # pip install torch-geometric torch-scatter torch-sparse

Usage:
    python SRC/preprocessing/evaluate_gnn_landmarks_loso.py

Expected runtime: ~40–90 min on a single GPU (LOSO = 67 folds).
"""

import os
import sys
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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── torch_geometric ────────────────────────────────────────────────────────────
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data  # kept for type hints; Batch no longer needed
except ImportError:
    print("ERROR: torch-geometric not installed.")
    print("  pip install torch-geometric")
    print("  See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# ── scipy for Delaunay triangulation (facial graph edges) ─────────────────────
from scipy.spatial import Delaunay as _Delaunay

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COORDS_NPZ  = os.path.join(BASE_DIR, "Results", "landmarks_gnn", "raw_coords.npz")
BIOSIG_CSV  = os.path.join(BASE_DIR, "Results", "biosignals_hrv", "all_67_hrv.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "gnn_loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS          = 35    # epochs per LOSO fold (35 = good convergence, fits in 1-hr SLURM limit)
BATCH_SIZE      = 64    # samples per batch (each sample = 24 frames = 24 graphs)
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.3
GAT_HIDDEN      = 32   # hidden channels per GAT head
GAT_HEADS       = 4    # attention heads in layer 1
EMB_DIM         = 32   # final embedding dimension per sample
SEED            = 42
OOF_META_SPLIT  = 0.0   # 0.0 = use in-sample probs (matches baseline methodology, faster)
                         # 0.20 = hold out 20% subjects for unbiased meta probs (slower)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD FACIAL GRAPH EDGES (Delaunay triangulation on mean face shape)
# ══════════════════════════════════════════════════════════════════════════════
# We build the graph AFTER loading coords (see Section 6) so we can use the
# actual BioVid mean face shape for the triangulation.
# EDGE_INDEX is set at the bottom of Section 6 once coords_all is loaded.

def build_edge_index_from_coords(coords_all):
    """
    Build a facial graph via Delaunay triangulation on the mean landmark
    positions across all BioVid samples.

    Why Delaunay instead of MediaPipe FACEMESH_TESSELATION:
      - MediaPipe 0.10.x removed the solutions API on Apple Silicon
      - Delaunay on the actual data mean face is anatomically grounded
      - Maximises minimum triangle angle → well-shaped, dense connectivity
      - ~1400–1600 undirected edges vs FACEMESH_TESSELATION's ~750

    Returns edge_index: LongTensor (2, 2*E) — bidirectional.
    """
    # Mean face: average over all samples, frames, subjects → (468, 2)
    mean_face = coords_all.reshape(-1, coords_all.shape[2], 2).mean(axis=0)

    tri    = _Delaunay(mean_face)
    edges  = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edges.add((min(a, b), max(a, b)))

    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GAT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FacialGAT(nn.Module):
    """
    2-layer Graph Attention Network on a facial landmark graph.

    Layer 1: GATConv(2 → hidden*heads)  with multi-head attention
    Layer 2: GATConv(hidden*heads → emb_dim)  aggregating to a single vector
    Pool   : Global mean pool → one (emb_dim,) vector per graph (= per frame)

    During a forward pass, frames from the same sample are batched together.
    The caller averages per-frame embeddings to get a per-sample embedding.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden: int = GAT_HIDDEN,
        heads: int = GAT_HEADS,
        emb_dim: int = EMB_DIM,
        num_classes: int = 2,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels, hidden,
            heads=heads, dropout=dropout, add_self_loops=True
        )
        self.bn1 = nn.BatchNorm1d(hidden * heads)

        self.gat2 = GATConv(
            hidden * heads, emb_dim,
            heads=1, concat=False, dropout=dropout, add_self_loops=True
        )
        self.bn2 = nn.BatchNorm1d(emb_dim)

        self.classifier = nn.Linear(emb_dim, num_classes)

    def embed(self, x, edge_index, batch):
        """
        Compute per-graph embeddings without the final classifier.

        Args:
            x          : (total_nodes, 2)    — landmark coords for all graphs in batch
            edge_index : (2, total_edges)    — re-indexed edges from Batch.from_data_list
            batch      : (total_nodes,)      — graph assignment for each node

        Returns:
            (num_graphs, emb_dim) — one embedding vector per graph
        """
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return global_mean_pool(x, batch)   # (num_graphs, emb_dim)

    def forward(self, x, edge_index, batch):
        emb = self.embed(x, edge_index, batch)
        return self.classifier(emb)         # (num_graphs, num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BATCHING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# GPU-side cache for batched edge_index + batch_vec.
# Since all graphs share the same topology (same 468 nodes, same 1365 edges),
# only the node features (x) change per batch.  We pre-build the structural
# tensors on GPU ONCE per unique n_graphs value and reuse them every batch.
# This eliminates a ~33 MB CPU→GPU transfer on every single mini-batch.
_GPU_STRUCT_CACHE: dict = {}

def _get_gpu_batch_structure(
    n_graphs: int,
    edge_index_cpu: torch.Tensor,
    device: torch.device,
    n_nodes: int = 468,
):
    """
    Return (edge_index_batched, batch_vec) already on `device`, cached.
    First call builds + uploads; subsequent calls are O(1) dict lookup.
    """
    key = (n_graphs, str(device))
    if key not in _GPU_STRUCT_CACHE:
        E       = edge_index_cpu.shape[1]
        offsets = (torch.arange(n_graphs) * n_nodes).repeat_interleave(E)
        ei_b    = edge_index_cpu.repeat(1, n_graphs) + offsets.unsqueeze(0)  # (2, n_graphs*E)
        bv      = torch.arange(n_graphs).repeat_interleave(n_nodes)          # (n_graphs*N,)
        _GPU_STRUCT_CACHE[key] = (ei_b.to(device), bv.to(device))
    return _GPU_STRUCT_CACHE[key]


def coords_to_tensors(
    coords_np: np.ndarray,
    edge_index_cpu: torch.Tensor,
    device: torch.device,
):
    """
    Fast vectorized batching for homogeneous graph batches.

    - Node features (x) are built from numpy and uploaded each call.
    - Edge structure (edge_index, batch_vec) is cached on GPU — uploaded once
      per unique batch size for the entire experiment.

    Args:
        coords_np     : (B, n_f, 468, 2) float32 numpy
        edge_index_cpu: (2, E) LongTensor on CPU
        device        : target torch.device

    Returns:
        x             : (B*n_f*468, 2) FloatTensor on device
        edge_index_b  : (2, B*n_f*E)  LongTensor  on device  [cached]
        batch_vec     : (B*n_f*468,)  LongTensor  on device  [cached]
        B, n_f        : ints
    """
    B, n_f, N, C = coords_np.shape
    n_graphs     = B * n_f
    x            = torch.tensor(
        coords_np.reshape(n_graphs * N, C), dtype=torch.float32
    ).to(device)
    edge_index_b, batch_vec = _get_gpu_batch_structure(n_graphs, edge_index_cpu, device, N)
    return x, edge_index_b, batch_vec, B, n_f


# ══════════════════════════════════════════════════════════════════════════════
# 4. NORMALISATION (person-specific, same strategy as baseline)
# ══════════════════════════════════════════════════════════════════════════════

def person_norm_train(coords, groups):
    """
    Normalise each training subject's coords by their own per-axis mean/std.
    coords : (N, 24, 468, 2)
    groups : (N,) — subject IDs
    """
    out = coords.copy()
    for sid in np.unique(groups):
        mask = groups == sid
        sub  = coords[mask]                           # (n_sub, 24, 468, 2)
        mean = sub.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, 2)
        std  = sub.std(axis=(0, 1, 2),  keepdims=True)
        std[std < 1e-8] = 1.0
        out[mask] = (coords[mask] - mean) / std
    return out


def person_norm_test(coords):
    """
    Normalise test subject's coords by their own mean/std.
    coords : (n_test, 24, 468, 2)
    """
    mean = coords.mean(axis=(0, 1, 2), keepdims=True)
    std  = coords.std(axis=(0, 1, 2),  keepdims=True)
    std[std < 1e-8] = 1.0
    return (coords - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def train_one_fold(coords_train, y_train, edge_index_cpu, device):
    """
    Train FacialGAT for one LOSO fold.

    Training strategy:
      - Each forward pass: flatten B samples × F frames → B*F graphs
      - Compute per-frame logits → average over frames → per-sample logit
      - Cross-entropy loss + CosineAnnealingLR scheduler

    Returns the trained model.
    """
    model     = FacialGAT().to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    N       = len(coords_train)
    indices = np.arange(N)

    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start: start + BATCH_SIZE]
            chunk_np  = coords_train[batch_idx]          # (b, F, 468, 2)
            labels    = torch.tensor(y_train[batch_idx], dtype=torch.long).to(device)

            x, edge_index_b, batch_vec, b, n_f = coords_to_tensors(chunk_np, edge_index_cpu, device)

            # Per-frame logits: (b*n_f, 2)
            frame_logits = model(x, edge_index_b, batch_vec)

            # Average logits across frames for each sample: (b, n_f, 2) → (b, 2)
            sample_logits = frame_logits.view(b, n_f, 2).mean(dim=1)

            loss = F.cross_entropy(sample_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

    return model


@torch.no_grad()
def predict_proba(model, coords, edge_index_cpu, device):
    """
    Compute softmax class probabilities for a set of samples.

    Args:
        coords : (N, 24, 468, 2) float32 numpy
    Returns:
        probs  : (N, 2) numpy — [P(PA2), P(PA3)]
    """
    model.eval()
    all_probs = []
    N         = len(coords)

    for start in range(0, N, BATCH_SIZE):
        chunk_np                            = coords[start: start + BATCH_SIZE]
        x, edge_index_b, batch_vec, b, n_f = coords_to_tensors(chunk_np, edge_index_cpu, device)

        frame_logits  = model(x, edge_index_b, batch_vec)
        sample_logits = frame_logits.view(b, n_f, 2).mean(dim=1)    # (b, 2)
        probs         = F.softmax(sample_logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)  # (N, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading raw coordinate data...")
if not os.path.exists(COORDS_NPZ):
    print(f"ERROR: {COORDS_NPZ} not found.")
    print("Run extract_landmarks_raw_coords.py first.")
    sys.exit(1)

npz          = np.load(COORDS_NPZ, allow_pickle=True)
coords_all   = npz["coords"].astype(np.float32)   # (N, 24, 468, 2)
labels_all   = npz["labels"].astype(np.int64)      # (N,)
groups_all   = npz["subject_ids"].astype(np.int64) # (N,)
snames_all   = npz["sample_names"]                 # (N,) str

print(f"  Loaded  : {len(labels_all)} samples, {np.unique(groups_all).size} subjects")
print(f"  PA2     : {(labels_all == 0).sum()}")
print(f"  PA3     : {(labels_all == 1).sum()}")

# ── Load biosignal features for stacked fusion comparison ─────────────────────
print("\nLoading biosignal features...")
bio_df   = pd.read_csv(BIOSIG_CSV)
EXCLUDE  = {"subject_id", "sample_name", "class_name", "label"}
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]

# Align biosignal samples to coords samples by sample_name
bio_lookup = {row["sample_name"]: row[bio_cols].values.astype(np.float32)
              for _, row in bio_df.iterrows()}

# Only keep coords samples that also have biosignals
valid_mask   = np.array([s in bio_lookup for s in snames_all])
coords_all   = coords_all[valid_mask]
labels_all   = labels_all[valid_mask]
groups_all   = groups_all[valid_mask]
snames_all   = snames_all[valid_mask]
bio_feats    = np.array([bio_lookup[s] for s in snames_all], dtype=np.float32)

print(f"  Matched : {len(labels_all)} samples after biosignal alignment")

# ── Per-frame geometric normalization ─────────────────────────────────────────
# Raw (x,y) coordinates encode head translation and camera distance.
# When a subject moves their head, ALL 468 landmarks shift together — this
# looks like "pain signal" to the GNN but is just head motion.
#
# Fix: for each frame independently,
#   1. subtract nose-tip  → removes translation (head left/right/up/down)
#   2. divide by IOD      → removes scale (camera distance, head size)
# This gives the GNN the same geometric invariances that the hand-engineered
# pairwise-distance features have baked in.
#
# Applied ONCE to all data here; person_norm is still applied inside LOSO folds.
NOSE_IDX      = 4    # MediaPipe FaceLandmarker nose tip
LEFT_EYE_IDX  = 33   # left eye outer corner
RIGHT_EYE_IDX = 263  # right eye outer corner

def frame_normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Per-frame geometric normalization for translation + scale invariance.

    coords : (..., n_frames, 468, 2) float32
    Returns: same shape, each frame's landmarks centered on nose tip
             and scaled by inter-ocular distance (IOD).
    """
    nose      = coords[..., NOSE_IDX:NOSE_IDX+1, :]               # (..., nf, 1, 2)
    left_eye  = coords[..., LEFT_EYE_IDX,  :]                     # (..., nf, 2)
    right_eye = coords[..., RIGHT_EYE_IDX, :]                     # (..., nf, 2)
    iod       = np.linalg.norm(right_eye - left_eye, axis=-1,
                               keepdims=True)[..., np.newaxis]     # (..., nf, 1, 1)
    iod       = np.maximum(iod, 1e-6)
    return (coords - nose) / iod

print("\nApplying per-frame geometric normalization (nose-centered, IOD-scaled)...")
coords_all = frame_normalize_coords(coords_all)
print(f"  Coords range after normalization: "
      f"[{coords_all.min():.2f}, {coords_all.max():.2f}]")

# ── Build facial graph from actual data mean face ─────────────────────────────
print("\nBuilding facial graph (Delaunay on BioVid mean face)...")
EDGE_INDEX        = build_edge_index_from_coords(coords_all)
EDGE_INDEX_DEVICE = EDGE_INDEX.to(DEVICE)
print(f"  Nodes : 468 landmarks")
print(f"  Edges : {EDGE_INDEX.shape[1] // 2} undirected edges")


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def bio_person_norm_train(X, groups):
    out = X.copy()
    for sid in np.unique(groups):
        mask       = groups == sid
        mean, std  = X[mask].mean(axis=0), X[mask].std(axis=0)
        std[std == 0] = 1.0
        out[mask]  = (X[mask] - mean) / std
    return out

def bio_person_norm_test(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


logo = LeaveOneGroupOut()
n_folds = np.unique(groups_all).size
print(f"\nRunning LOSO ({n_folds} folds)...")
print("=" * 60)

gnn_accs     = []   # GNN landmarks only
stacked_accs = []   # GNN probs + biosignal RF probs → LogReg

fold = 0
for train_idx, test_idx in logo.split(coords_all, labels_all, groups_all):
    fold += 1
    g_train = groups_all[train_idx]
    y_train = labels_all[train_idx]
    y_test  = labels_all[test_idx]

    # ── GNN landmark features: person-specific normalisation ─────────────────
    c_train = person_norm_train(coords_all[train_idx], g_train)
    c_test  = person_norm_test(coords_all[test_idx])

    # ── Biosignal features: person-specific normalisation ────────────────────
    b_train = bio_person_norm_train(bio_feats[train_idx], g_train)
    b_test  = bio_person_norm_test(bio_feats[test_idx])

    # ── Subject-stratified split: base (80%) trains models, meta (20%) trains meta-learner ──
    # Using in-sample probs would give the GNN/RF perfect-looking train probs,
    # biasing the meta-learner toward whichever model is more overconfident rather
    # than whichever generalises better. Holdout subjects = unbiased calibration.
    rng             = np.random.default_rng(SEED + fold)   # fold-specific seed → stable splits
    train_subjects  = np.unique(g_train)
    n_meta_subs     = max(1, int(len(train_subjects) * OOF_META_SPLIT))
    meta_subs       = rng.choice(train_subjects, size=n_meta_subs, replace=False)
    base_mask       = ~np.isin(g_train, meta_subs)         # ~80% of train samples
    meta_mask       =  np.isin(g_train, meta_subs)         # ~20% of train samples

    c_base, c_meta  = c_train[base_mask],  c_train[meta_mask]
    b_base, b_meta  = b_train[base_mask],  b_train[meta_mask]
    y_base, y_meta  = y_train[base_mask],  y_train[meta_mask]

    # ── Step 1: train on base subset → get unbiased probs on meta subset ─────
    if OOF_META_SPLIT > 0:
        model_base = train_one_fold(c_base, y_base, EDGE_INDEX.cpu(), DEVICE)
        rf_base    = RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_split=10,
            max_features="sqrt", random_state=SEED, n_jobs=-1
        )
        rf_base.fit(b_base, y_base)

        gnn_meta_probs = predict_proba(model_base, c_meta, EDGE_INDEX.cpu(), DEVICE)
        bio_meta_probs = rf_base.predict_proba(b_meta)

        X_meta_train = np.hstack([gnn_meta_probs, bio_meta_probs])   # (n_meta, 4)
        meta         = LogisticRegression(max_iter=1000, random_state=SEED)
        meta.fit(X_meta_train, y_meta)

        del model_base, rf_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Step 2: retrain on FULL training set → final test inference ──────────
    model  = train_one_fold(c_train, y_train, EDGE_INDEX.cpu(), DEVICE)
    rf_bio = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_split=10,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf_bio.fit(b_train, y_train)

    gnn_test_probs = predict_proba(model, c_test, EDGE_INDEX.cpu(), DEVICE)   # (n_test, 2)
    bio_test_probs = rf_bio.predict_proba(b_test)                              # (n_test, 2)

    # ── GNN-only accuracy (full model) ────────────────────────────────────────
    gnn_acc = accuracy_score(y_test, gnn_test_probs.argmax(axis=1))
    gnn_accs.append(gnn_acc)

    # ── Stacked fusion: meta-learner on unbiased probs ────────────────────────
    if OOF_META_SPLIT > 0:
        X_meta_test = np.hstack([gnn_test_probs, bio_test_probs])              # (n_test, 4)
        stacked_acc = accuracy_score(y_test, meta.predict(X_meta_test))
    else:
        # Fallback: in-sample probs (matches original baseline methodology)
        gnn_train_probs = predict_proba(model, c_train, EDGE_INDEX.cpu(), DEVICE)
        bio_train_probs = rf_bio.predict_proba(b_train)
        meta_fb         = LogisticRegression(max_iter=1000, random_state=SEED)
        meta_fb.fit(np.hstack([gnn_train_probs, bio_train_probs]), y_train)
        stacked_acc     = accuracy_score(y_test, meta_fb.predict(
            np.hstack([gnn_test_probs, bio_test_probs])))

    stacked_accs.append(stacked_acc)

    print(f"  Fold {fold:2d}/{n_folds} | "
          f"GNN: {gnn_acc*100:.1f}% | "
          f"Stacked: {stacked_acc*100:.1f}% | "
          f"Running GNN mean: {np.mean(gnn_accs)*100:.1f}%")

    # Free GPU memory between folds
    del model, rf_bio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESULTS
# ══════════════════════════════════════════════════════════════════════════════

gnn_mean,     gnn_std     = np.mean(gnn_accs)     * 100, np.std(gnn_accs)     * 100
stacked_mean, stacked_std = np.mean(stacked_accs) * 100, np.std(stacked_accs) * 100

print(f"\n{'=' * 60}")
print(f"  GNN LANDMARK LOSO RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline: RF landmarks (flat stats)  61.4% ± 13.1%  [from baseline]")
print(f"  Baseline: Stacked fusion (RF+RF+LR)  65.3% ± 14.1%  [from baseline]")
print(f"{'=' * 60}")
print(f"  GNN landmarks only     :  {gnn_mean:.1f}% ± {gnn_std:.1f}%")
print(f"  GNN + biosignal RF     :  {stacked_mean:.1f}% ± {stacked_std:.1f}%")
print(f"{'=' * 60}")
print(f"  Δ GNN vs RF landmarks  :  {gnn_mean - 61.4:+.1f}%")
print(f"  Δ Stacked vs baseline  :  {stacked_mean - 65.3:+.1f}%")
print(f"{'=' * 60}")

# Save per-fold results
results_df = pd.DataFrame({
    "fold":         range(1, n_folds + 1),
    "subject_id":   np.unique(groups_all),
    "gnn_acc":      gnn_accs,
    "stacked_acc":  stacked_accs,
})
out_csv = os.path.join(RESULTS_DIR, "gnn_loso_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nPer-fold results saved: {out_csv}")

# Summary
summary = {
    "gnn_only_mean":     round(gnn_mean,     2),
    "gnn_only_std":      round(gnn_std,      2),
    "gnn_stacked_mean":  round(stacked_mean, 2),
    "gnn_stacked_std":   round(stacked_std,  2),
    "baseline_rf_lm":    61.4,
    "baseline_stacked":  65.3,
    "delta_gnn_vs_rf_lm":    round(gnn_mean     - 61.4, 2),
    "delta_stacked_vs_base": round(stacked_mean - 65.3, 2),
}
summary_df = pd.DataFrame([summary])
out_summary = os.path.join(RESULTS_DIR, "gnn_loso_summary.csv")
summary_df.to_csv(out_summary, index=False)
print(f"Summary saved        : {out_summary}")
