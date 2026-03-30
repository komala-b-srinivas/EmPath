import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import neurokit2 as nk

BIOVID_ROOT = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw"
SAMPLES_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/starting_point/samples.csv"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals_hrv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 512
SIGNAL_COLS = ["gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"]

EXCLUDED_SUBJECTS = {
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
}

def compute_baseline_per_subject(subject_name):
    bl_dir = os.path.join(BIOVID_ROOT, "biosignals_filtered", subject_name)
    if not os.path.exists(bl_dir):
        return None
    bl_signals = {col: [] for col in SIGNAL_COLS}
    for fname in os.listdir(bl_dir):
        if "BL1" in fname and fname.endswith("_bio.csv"):
            try:
                df = pd.read_csv(os.path.join(bl_dir, fname), sep="\t")
                for col in SIGNAL_COLS:
                    if col in df.columns:
                        bl_signals[col].append(df[col].values)
            except:
                continue
    baselines = {}
    for col in SIGNAL_COLS:
        if len(bl_signals[col]) > 0:
            min_len = min(len(s) for s in bl_signals[col])
            baselines[col] = np.mean(
                [s[:min_len] for s in bl_signals[col]], axis=0)
        else:
            baselines[col] = None
    return baselines

def compute_similarity(signal_arr, baseline_arr):
    if baseline_arr is None or len(signal_arr) == 0:
        return 0.0, 0.0
    min_len = min(len(signal_arr), len(baseline_arr))
    s, b = signal_arr[:min_len], baseline_arr[:min_len]
    try:
        corr, _ = pearsonr(s, b)
        corr = 0.0 if np.isnan(corr) else corr
    except:
        corr = 0.0
    try:
        hist_2d, _, _ = np.histogram2d(s, b, bins=10)
        pxy   = hist_2d / (hist_2d.sum() + 1e-10)
        px    = pxy.sum(axis=1)
        py    = pxy.sum(axis=0)
        px_py = px[:, None] * py[None, :]
        mask  = pxy > 0
        mi    = np.sum(pxy[mask] * np.log(
            pxy[mask] / (px_py[mask] + 1e-10)))
    except:
        mi = 0.0
    return corr, mi

def compute_entropy(arr):
    try:
        hist, _ = np.histogram(arr, bins=20)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    except:
        return 0.0

def extract_features(bio_path, baselines):
    df   = pd.read_csv(bio_path, sep="\t")
    gsr  = df["gsr"].values
    ecg  = df["ecg"].values
    trap = df["emg_trapezius"].values
    corr = df["emg_corrugator"].values
    zyg  = df["emg_zygomaticus"].values

    feats = {}
    feats.update({
        "gsr_mean":      np.mean(gsr),
        "gsr_std":       np.std(gsr),
        "gsr_slope":     np.polyfit(np.arange(len(gsr)), gsr, 1)[0],
        "ecg_mean":      np.mean(ecg),
        "ecg_std":       np.std(ecg),
        "ecg_max":       np.max(ecg),
        "emg_trap_mean": np.mean(np.abs(trap)),
        "emg_trap_std":  np.std(trap),
        "emg_corr_mean": np.mean(np.abs(corr)),
        "emg_corr_std":  np.std(corr),
        "emg_corr_max":  np.max(np.abs(corr)),
        "emg_zyg_mean":  np.mean(np.abs(zyg)),
        "emg_zyg_std":   np.std(zyg),
    })

    for name, arr in [("gsr", gsr), ("ecg", ecg),
                      ("emg_corr", corr), ("emg_zyg", zyg)]:
        feats[f"{name}_shannon"] = compute_entropy(arr)

    for col_name, arr in [("gsr", gsr), ("ecg", ecg),
                           ("emg_trap", trap), ("emg_corr", corr),
                           ("emg_zyg",  zyg)]:
        bl_key = col_name.replace("emg_trap", "emg_trapezius")\
                         .replace("emg_corr", "emg_corrugator")\
                         .replace("emg_zyg",  "emg_zygomaticus")
        bl = baselines.get(bl_key) if baselines else None
        c, mi = compute_similarity(arr, bl)
        feats[f"{col_name}_sim_corr"] = c
        feats[f"{col_name}_sim_mi"]   = mi

    feats["gsr_x_ecg"] = np.mean(gsr) * np.mean(ecg)
    feats["emg_asymmetry"] = abs(
        np.mean(np.abs(trap)) - np.mean(np.abs(corr))) / \
        (np.mean(np.abs(trap)) + np.mean(np.abs(corr)) + 1e-6)

    try:
        ecg_signals, info = nk.ecg_process(ecg, sampling_rate=FS)
        hrv_time = nk.hrv_time(ecg_signals, sampling_rate=FS, show=False)
        feats["hrv_pnn50"]  = float(hrv_time["HRV_pNN50"].values[0]) \
                              if "HRV_pNN50"  in hrv_time.columns else 0.0
        feats["hrv_rmssd"]  = float(hrv_time["HRV_RMSSD"].values[0]) \
                              if "HRV_RMSSD"  in hrv_time.columns else 0.0
        feats["hrv_sdnn"]   = float(hrv_time["HRV_SDNN"].values[0])  \
                              if "HRV_SDNN"   in hrv_time.columns else 0.0
        feats["hrv_meannn"] = float(hrv_time["HRV_MeanNN"].values[0])\
                              if "HRV_MeanNN" in hrv_time.columns else 0.0
        hrv_freq = nk.hrv_frequency(ecg_signals, sampling_rate=FS, show=False)
        lf = float(hrv_freq["HRV_LF"].values[0]) \
             if "HRV_LF" in hrv_freq.columns else 1.0
        hf = float(hrv_freq["HRV_HF"].values[0]) \
             if "HRV_HF" in hrv_freq.columns else 1.0
        feats["hrv_lf_hf_ratio"] = lf / (hf + 1e-6)
        mean_hr = 60000 / (feats["hrv_meannn"] + 1e-6)
        feats["parasympathetic_tone"] = feats["hrv_rmssd"] / (mean_hr + 1e-6)
    except:
        for key in ["hrv_pnn50", "hrv_rmssd", "hrv_sdnn",
                    "hrv_meannn", "hrv_lf_hf_ratio", "parasympathetic_tone"]:
            feats[key] = 0.0

    return feats

# ── Load samples ───────────────────────────────────────────────────────
print("Loading samples...")
samples_df = pd.read_csv(SAMPLES_CSV, sep="\t")
pa_samples = samples_df[samples_df["class_name"].isin(["PA2", "PA3"])]
pa_samples = pa_samples[~pa_samples["subject_name"].isin(EXCLUDED_SUBJECTS)]
print(f"Subjects after exclusion: {pa_samples['subject_id'].nunique()}")

print("Computing baselines...")
all_baselines = {}
for _, row in samples_df[["subject_id", "subject_name"]]\
        .drop_duplicates().iterrows():
    if row["subject_name"] in EXCLUDED_SUBJECTS:
        continue
    bl = compute_baseline_per_subject(row["subject_name"])
    if bl:
        all_baselines[str(row["subject_id"])] = bl
print(f"Baselines for {len(all_baselines)} subjects ✓")

print("Extracting features...")
rows   = []
failed = 0

for _, row in tqdm(pa_samples.iterrows(), total=len(pa_samples)):
    bio_path = os.path.join(
        BIOVID_ROOT, "biosignals_filtered",
        row["subject_name"],
        row["sample_name"] + "_bio.csv"
    )
    if not os.path.exists(bio_path):
        failed += 1
        continue
    try:
        baselines = all_baselines.get(str(row["subject_id"]), {})
        feats     = extract_features(bio_path, baselines)
        feats["subject_id"]  = row["subject_id"]
        feats["sample_name"] = row["sample_name"]
        feats["class_name"]  = row["class_name"]
        feats["label"]       = 0 if row["class_name"] == "PA2" else 1
        rows.append(feats)
    except:
        failed += 1

print(f"Extracted: {len(rows)} samples, {failed} failed")

all_df   = pd.DataFrame(rows)
EXCLUDE  = ["subject_id", "sample_name", "class_name", "label"]
bio_cols = [c for c in all_df.columns if c not in EXCLUDE]
X        = np.nan_to_num(all_df[bio_cols].values)
y        = all_df["label"].values
groups   = all_df["subject_id"].values

print(f"Features per sample: {len(bio_cols)}")

logo = LeaveOneGroupOut()
accs = []
fold = 0

print("Running LOSO with person-specific normalization...")
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
print(f"  LOSO + PERSON-SPECIFIC NORMALIZATION")
print(f"{'='*55}")
print(f"  Mean accuracy: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
print(f"{'='*55}")

all_df.to_csv(os.path.join(OUTPUT_DIR, "all_67_hrv.csv"), index=False)
print("\nSaved ✓")