import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

BIOSIG_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Results/biosignals_hrv/all_67_hrv.csv"

print("Loading data...")
bio_df = pd.read_csv(BIOSIG_CSV)

EXCLUDE  = ["subject_id", "sample_name", "class_name", "label"]
bio_cols = [c for c in bio_df.columns if c not in EXCLUDE]

X      = np.nan_to_num(bio_df[bio_cols].values)
y      = bio_df["label"].values
groups = bio_df["subject_id"].values

print(f"Samples:  {len(bio_df)}")
print(f"Subjects: {bio_df['subject_id'].nunique()}")
print(f"Features: {len(bio_cols)}")

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

print("\nRunning MLP LOSO with person-specific normalization...")
for train_idx, test_idx in logo.split(X, y, groups):
    fold += 1
    groups_train = groups[train_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]

    X_train_norm = person_norm_train(X[train_idx], groups_train)
    X_test_norm  = person_norm_test(X[test_idx])

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train_norm, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_norm))
    accs.append(acc)

    if fold % 10 == 0:
        print(f"  Fold {fold} | Running mean: {np.mean(accs)*100:.1f}%")

print(f"\n{'='*55}")
print(f"  MLP LOSO RESULTS")
print(f"{'='*55}")
print(f"  MLP random split (old)  : 51.2%")
print(f"  RF LOSO + person-norm   : 63.1% ± 11.6%")
print(f"  MLP LOSO + person-norm  : {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
print(f"{'='*55}")