# EmPath v2 — Complete Pipeline Technical Guide
**From Raw Dataset to Final Model: Step-by-Step**

---

## Overview

EmPath classifies pain intensity between two adjacent heat stimulus levels —
PA2 (~43°C) and PA3 (~45°C) — using two modalities:
- **Biosignals:** GSR, ECG, EMG (physiological)
- **Facial Video:** MediaPipe landmark geometry (visual)

Both are fused via a stacked generalization architecture and evaluated under
Leave-One-Subject-Out (LOSO) cross-validation on 67 reactive subjects.

---

## STAGE 1 — Dataset

### What is BioVid?
The **BioVid Heat Pain Database** contains synchronized recordings of 87
subjects exposed to calibrated heat stimuli on the forearm. Each subject
received stimuli at 5 pain levels (BL1=baseline, PA1–PA4=increasing pain).

| Property | Value |
|---|---|
| Total subjects | 87 |
| Reactive subjects used | 67 |
| Excluded (non-reactive) | 20 |
| Pain levels targeted | PA2, PA3 |
| Samples per subject | ~40 |
| Total samples | ~2,680 |
| Biosignal sampling rate | 512 Hz |
| Video frame rate | 25 fps |
| Window duration | 5.5 seconds |

### File structure on disk
```
Data/Raw/
├── biosignals_filtered/
│   └── <subject_name>/
│       └── <sample_name>_bio.csv    ← TSV, 5 columns: gsr ecg emg_trapezius
│                                        emg_corrugator emg_zygomaticus
├── video/
│   └── <subject_name>/
│       └── <sample_name>.mp4        ← 5.5s facial video clip
└── starting_point/
    └── samples.csv                  ← master index (tab-separated)
                                        columns: subject_id, subject_name,
                                        sample_name, class_name
```

### Subject naming convention
```
082315_w_60  →  MMDDYY _ gender _ age
```

### Why 67 not 87?
20 subjects show flat physiological responses to heat — their GSR, ECG, and
EMG signals do not change between baseline and pain stimulus. Including them
forces the model to learn from noise. These "non-reactive" or "stoic" subjects
are excluded per Werner & Al-Hamadi (2017) recommendation.

```python
EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}
```

---

## STAGE 2 — Biosignal Feature Extraction

**Script:** `SRC/preprocessing/extract_biosignals_all87.py`
**Output:** `Results/biosignals_hrv/all_67_hrv.csv`

### What signals are available?
Each TSV biosignal file contains 5 channels at 512 Hz:

| Column | Signal | What it measures |
|---|---|---|
| `gsr` | Galvanic Skin Response | Electrodermal activity — skin conductance |
| `ecg` | Electrocardiogram | Heart electrical activity |
| `emg_trapezius` | Trapezius EMG | Shoulder/neck muscle tension |
| `emg_corrugator` | Corrugator EMG | Brow furrow muscle |
| `emg_zygomaticus` | Zygomaticus EMG | Cheek/smile muscle |

### Step 1 — Compute per-subject baseline
Before extracting pain features, compute each subject's resting baseline
from their BL1 (no-stimulus) recordings:

```python
def compute_baseline_per_subject(subject_name):
    # Load all BL1 (baseline) files for this subject
    for fname in os.listdir(bl_dir):
        if "BL1" in fname and fname.endswith("_bio.csv"):
            df = pd.read_csv(fname, sep="\t")
            bl_signals[col].append(df[col].values)
    # Average across all baseline trials → one baseline signal per channel
    baselines[col] = np.mean([s[:min_len] for s in bl_signals[col]], axis=0)
```

This baseline is later used to compute correlation and mutual information
between pain signal and resting state.

### Step 2 — Extract 35 features per sample
For each 5.5-second pain window:

```python
def extract_features(bio_path, baselines):
    gsr  = df["gsr"].values          # shape: (2816,) at 512 Hz × 5.5s
    ecg  = df["ecg"].values
    trap = df["emg_trapezius"].values

    # GSR features (skin conductance)
    feats["gsr_mean"]    = np.mean(gsr)
    feats["gsr_std"]     = np.std(gsr)
    feats["gsr_slope"]   = np.polyfit(np.arange(len(gsr)), gsr, 1)[0]
    feats["gsr_entropy"] = compute_entropy(gsr)

    # ECG features (heart)
    feats["ecg_mean"]    = np.mean(ecg)
    feats["ecg_std"]     = np.std(ecg)
    feats["ecg_max"]     = np.max(ecg)

    # EMG features (muscles)
    feats["emg_trap_mean"] = np.mean(np.abs(trap))
    feats["emg_trap_std"]  = np.std(trap)
    ...
    # Baseline similarity (pain vs rest comparison)
    feats["gsr_corr_bl"], feats["gsr_mi_bl"] = compute_similarity(gsr, baselines["gsr"])
    # HRV via NeuroKit2
    hrv = nk.hrv(nk.ecg_peaks(ecg, sampling_rate=512)[1], sampling_rate=512)
    feats["hrv_meanNN"] = hrv["HRV_MeanNN"].values[0]
    feats["hrv_sdnn"]   = hrv["HRV_SDNN"].values[0]
```

### Shannon entropy — what it means
```python
def compute_entropy(arr):
    hist, _ = np.histogram(arr, bins=20)
    hist = hist / hist.sum()             # normalize to probability
    hist = hist[hist > 0]               # remove zero bins
    return -np.sum(hist * np.log(hist)) # Shannon entropy formula
```
**Pain windows are more irregular (higher entropy) than baseline.**
A flat resting GSR has low entropy. A spiking pain GSR has high entropy.
The model uses this irregularity as a pain discriminator.

### gsr_slope — the dominant feature
```python
feats["gsr_slope"] = np.polyfit(np.arange(len(gsr)), gsr, 1)[0]
```
Fits a degree-1 polynomial (straight line) to the GSR signal.
The slope (coefficient) is positive when GSR is rising (pain arousal),
negative when falling. SHAP later confirms this as the #1 feature with
mean |SHAP| = 0.0821, 3.4× more important than the next feature.

### Output
`all_67_hrv.csv` — shape: (2680, 39)
- 35 feature columns
- subject_id, sample_name, class_name, label

---

## STAGE 3 — Facial Landmark Feature Extraction

**Script:** `SRC/preprocessing/extract_landmarks_all67.py`
**Output:** `Results/landmarks_all67/landmarks_all67.csv`

### MediaPipe FaceMesh setup
```python
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,    # treat each frame independently
    max_num_faces=1,
    refine_landmarks=True,     # enables iris + lips refinement (478 pts total)
    min_detection_confidence=0.5
)
```

### Step 1 — Sample 24 evenly spaced frames from the video
```python
def extract_video_features(video_path, num_frames=24):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ~137 frames at 25fps × 5.5s
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    # → frames at t=0, 0.24s, 0.48s, ... 5.28s
```
24 frames is enough to capture peak expression while keeping compute low.

### Step 2 — Extract landmark distances per frame
```python
# Map 468 (x,y) points onto pixel coordinates
lm  = face_mesh.process(rgb).multi_face_landmarks[0].landmark
pts = {i: (lm[i].x * w, lm[i].y * h) for i in range(len(lm))}

# Key landmark index groups
LEFT_BROW  = [70, 63, 105, 66, 107]   # left eyebrow points
RIGHT_BROW = [336, 296, 334, 293, 300]
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH      = [61, 291, 0, 17, 78, 308]

# Compute geometric distances
features["brow_furrow"]         = dist(pts[LEFT_BROW[0]], pts[RIGHT_BROW[0]])
features["left_brow_eye_dist"]  = dist(brow_center_L, eye_center_L)
features["mouth_aspect_ratio"]  = mouth_height / (mouth_width + 1e-6)
features["avg_eye_openness"]    = (left_eye_open + right_eye_open) / 2

# Scale invariance: normalize all distances by face width
face_width = dist(pts[234], pts[454])  # cheek-to-cheek
for k in features:
    features[k] = features[k] / (face_width + 1e-6)
```

Dividing by face width makes features camera-distance invariant.
A subject sitting 50cm away vs 80cm away produces identical feature values.

### Step 3 — Aggregate across 24 frames
```python
for k in keys:
    vals = [f[k] for f in all_features]       # 24 values per feature
    result[f"{k}_mean"] = np.mean(vals)        # average expression state
    result[f"{k}_std"]  = np.std(vals)         # expression variability
```
**Why mean AND std?**
- `mean` captures the average facial state (e.g., average brow height)
- `std` captures how much the expression changed during the window
  (high std = dynamic expression, low std = frozen face)

11 raw distances × 2 (mean + std) = **22 landmark features per sample**

### Output
`landmarks_all67.csv` — shape: (2680, 26)
- 22 feature columns
- subject_id, sample_name, class_name, label

---

## STAGE 4 — Person-Specific Normalization

**Used in:** `evaluate_stacked_fusion_loso.py` (inside LOSO loop)

### The problem with global normalization
Person A has resting GSR = 10 µS. Person B has resting GSR = 2 µS.
Person A's PA2 value = 12. Person B's PA3 value = 11.
Global normalization: 12 > 11 → model incorrectly ranks A's PA2 higher than B's PA3.
The model learns *who the person is*, not *how much pain they feel*.

### The fix — z-score per person
```python
def person_norm_train(X, groups):
    X_norm = X.copy()
    for sid in np.unique(groups):          # iterate each training subject
        mask = groups == sid
        mean = X[mask].mean(axis=0)        # this subject's mean per feature
        std  = X[mask].std(axis=0)
        std[std == 0] = 1                  # avoid divide-by-zero
        X_norm[mask] = (X[mask] - mean) / std
    return X_norm
```

After normalization, every subject's features are relative to their own mean.
A value of +2.0 means "2 standard deviations above THIS person's normal."
The model now learns pain patterns, not person identity.

### Test subject normalization (no leakage)
```python
def person_norm_test(X_test):
    # Cannot use training stats — that would leak info about training subjects
    mean = X_test.mean(axis=0)    # test subject's own mean
    std  = X_test.std(axis=0)
    return (X_test - mean) / std
```

**Impact: +3.2% accuracy** (63.1% without → 65.3% with normalization on stacked fusion)

---

## STAGE 5 — LOSO Cross-Validation Setup

**Leave-One-Subject-Out** is the gold standard for subject-independent models.

```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()

# groups = array of subject IDs, one per sample
# Each fold: train on 66 subjects, test on 1 completely held-out subject
for train_idx, test_idx in logo.split(X, y, groups):
    # train_idx: all samples from 66 subjects
    # test_idx:  all samples from 1 subject (never seen during training)
```

### Why LOSO matters
Random split (80/20): samples from the same subject appear in both train and
test. The model memorizes individual physiology → inflated accuracy.
LOSO: the test subject is completely new. The model must generalize to a
person it has never seen. This is the clinically relevant scenario.

| Split type | What model learns | Accuracy |
|---|---|---|
| Random 80/20 | Subject identity | ~75–85% (inflated) |
| LOSO | Pain patterns | 65.3% (honest) |

---

## STAGE 6 — Base Model Training (Random Forest × 2)

```python
rf_bio = RandomForestClassifier(
    n_estimators=300,       # 300 trees
    max_depth=4,            # shallow — prevents overfitting at small N
    min_samples_split=10,   # at least 10 samples to split a node
    max_features='sqrt',    # √35 ≈ 6 features per split (randomness)
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf_bio.fit(X_bio_train, y_train)   # 35 biosignal features, 66 subjects
rf_lm.fit(X_lm_train,  y_train)   # 22 landmark features, 66 subjects
```

### Why Random Forest over deep learning?
Tested 13 deep/foundation model variants — all underperformed:

| Model | Accuracy |
|---|---|
| TCN (deep) | 55.9% |
| MLP (deep) | 51.2% |
| BIOT (foundation) | 54.4% |
| PainFormer (foundation) | 53.1% |
| GNN (graph neural net) | 51.7% |
| **RF (this work)** | **63.1%** |

With only 2,680 samples and 67 subjects under LOSO, deep models overfit.
RF is shallow, regularized, and empirically correct at this scale.

### Why max_depth=4?
Each LOSO fold trains on ~2,640 samples (66 subjects × ~40 each).
Deep trees would memorize training patterns that don't generalize.
Depth 4 = at most 16 leaf nodes per tree → forced generalization.

---

## STAGE 7 — Stacked Generalization (Meta-Learner)

**Core idea:** Instead of concatenating raw features (early fusion), use
each RF's calibrated probability output as input to a meta-learner.

```python
# Get probability outputs from each base model
bio_train_probs = rf_bio.predict_proba(X_bio_train)  # shape: (N, 2) → [P(PA2), P(PA3)]
lm_train_probs  = rf_lm.predict_proba(X_lm_train)   # shape: (N, 2)
bio_test_probs  = rf_bio.predict_proba(X_bio_test)
lm_test_probs   = rf_lm.predict_proba(X_lm_test)

# Stack into 4-feature meta input
X_meta_train = np.hstack([bio_train_probs, lm_train_probs])  # shape: (N, 4)
X_meta_test  = np.hstack([bio_test_probs,  lm_test_probs])

# Meta-learner: learns how much to trust each modality
meta = LogisticRegression(random_state=42, max_iter=1000)
meta.fit(X_meta_train, y_train)
y_pred = meta.predict(X_meta_test)
```

### Why probabilities, not raw features?
Raw features from biosignals (GSR in µS) and landmarks (normalized ratios)
are on completely different scales. Concatenating them directly gives the
meta-learner inconsistent input.

Probabilities are always on [0, 1] and already encode each model's
confidence. The meta-learner sees: "biosignal says PA3 with 78% confidence,
landmarks say PA3 with 65% confidence" and learns the optimal combination.

### What the LogReg meta-learner learns
The 4 inputs are: [P_bio(PA2), P_bio(PA3), P_lm(PA2), P_lm(PA3)]
LogReg fits weights w1, w2, w3, w4 such that:
```
P(PA3) = sigmoid(w1·P_bio(PA2) + w2·P_bio(PA3) + w3·P_lm(PA2) + w4·P_lm(PA3) + b)
```
If biosignals are more reliable for a subject, w2 and w4 will be larger.
If landmarks are more reliable, w3 and w4 will dominate.

### Fusion comparison
| Method | Accuracy |
|---|---|
| Biosignal RF alone | 63.1% |
| Landmark RF alone | 61.4% |
| Early fusion (concat → RF) | 64.6% |
| **Stacked fusion (this work)** | **65.3%** |

---

## STAGE 8 — SHAP Explainability

**Script:** `SRC/preprocessing/shap_analysis_loso.py`
**Output:** `Results/error_analysis_v2/`

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
score for each individual prediction. Uses TreeExplainer for exact values.

```python
import shap
explainer = shap.TreeExplainer(rf_bio)
shap_values = explainer.shap_values(X_bio_test)
# shap_values[1] → SHAP values for class PA3
# shape: (n_test_samples, 35)
# positive = pushes toward PA3, negative = pushes toward PA2
```

### Key finding
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | gsr_slope | 0.0821 |
| 2 | ecg_std | 0.0241 |
| 3 | gsr_std | 0.0198 |
| 4 | hrv_sdnn | 0.0187 |
| 5 | emg_trap_std | 0.0156 |

`gsr_slope` dominates at 3.4× the next feature. Rising skin conductance
during a pain window is the strongest single indicator of PA3 vs PA2.

### Out-of-fold SHAP (no leakage)
SHAP is computed on the test subject in each fold — the model has never
seen this subject during training. Every SHAP value is leak-free.

---

## STAGE 9 — Final Results

### Aggregate across all 67 LOSO folds
```python
accs = []
for train_idx, test_idx in logo.split(...):
    # ... full pipeline per fold ...
    acc = accuracy_score(y_test, meta.predict(X_meta_test))
    accs.append(acc)

print(f"Mean: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
# → 65.3% ± 14.1%
```

### What the ±14.1% means
Individual subjects range from ~44% (below chance — model cannot generalize
to this subject's physiology) to ~88% (excellent generalization). The high
std reflects real inter-subject variability, not a flaw in the model.
Some subjects' pain response is simply unlike any subject in the training set.

### Final metrics
| Metric | Value |
|---|---|
| LOSO Accuracy | 65.3% ± 14.1% |
| AUC | 0.719 |
| F1-Score | 0.653 |
| Confusion matrix (PA2→PA2) | 884 correct |
| Confusion matrix (PA3→PA3) | 870 correct |
| PA2→PA3 errors | 456 |
| PA3→PA2 errors | 470 |

Errors are balanced between both classes — the model does not systematically
over-predict one pain level. This is important for clinical use.

---

## How to Run the Full Pipeline

```bash
# Step 1 — Extract biosignal features (~5 min, CPU)
python SRC/preprocessing/extract_biosignals_all87.py

# Step 2 — Extract landmark features (~45 min, CPU)
python SRC/preprocessing/extract_landmarks_all67.py

# Step 3 — Run stacked fusion LOSO (~2 min)
python SRC/preprocessing/evaluate_stacked_fusion_loso.py

# Step 4 — Run SHAP analysis (~10 min)
python SRC/preprocessing/shap_analysis_loso.py

# Step 5 — Launch Streamlit demo
streamlit run app.py
```

### Dependencies
```
scikit-learn==1.7.2   # pinned — empath_model.pkl saved with this version
neurokit2             # HRV extraction from ECG
mediapipe             # FaceMesh landmark extraction
opencv-python         # video frame extraction
shap                  # TreeExplainer for interpretability
streamlit             # demo app
pandas, numpy, scipy, matplotlib, seaborn
```

---

## File Map — What Produces What

```
extract_biosignals_all87.py   →  Results/biosignals_hrv/all_67_hrv.csv
extract_landmarks_all67.py    →  Results/landmarks_all67/landmarks_all67.csv
evaluate_stacked_fusion_loso.py →  console output (accuracy, F1, confusion)
shap_analysis_loso.py         →  Results/error_analysis_v2/*.png + *.csv
save_final_model.py           →  Models/empath_model.pkl
save_signals_plot.py          →  Models/signal_plots/*.png
app.py                        →  Streamlit app at localhost:8501
```
