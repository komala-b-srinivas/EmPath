import pandas as pd
import numpy as np
import os

SAMPLES_CSV = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/Raw/starting_point/samples.csv"
OUTPUT_DIR  = "/Users/komalabelursrinivas/Desktop/Capstone/EmPath/Data/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading samples...")
df = pd.read_csv(SAMPLES_CSV, sep="\t")
print(f"Total samples: {len(df)}")
print(f"Classes: {df['class_name'].unique()}")
print(f"Subjects: {df['subject_id'].nunique()}")

# ── Select 50 subjects for initial experiments ─────────────────────────
pa_df    = df[df["class_name"].isin(["PA2", "PA3"])]
subjects = pa_df["subject_id"].unique()
np.random.seed(42)
np.random.shuffle(subjects)

train_subjects = subjects[:35]
val_subjects   = subjects[35:42]
test_subjects  = subjects[42:50]

train_df = pa_df[pa_df["subject_id"].isin(train_subjects)]
val_df   = pa_df[pa_df["subject_id"].isin(val_subjects)]
test_df  = pa_df[pa_df["subject_id"].isin(test_subjects)]

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), sep="\t", index=False)
val_df.to_csv(  os.path.join(OUTPUT_DIR, "val.csv"),   sep="\t", index=False)
test_df.to_csv( os.path.join(OUTPUT_DIR, "test.csv"),  sep="\t", index=False)

print(f"\nSplits saved:")
print(f"  Train: {len(train_df)} samples ({len(train_subjects)} subjects)")
print(f"  Val:   {len(val_df)} samples ({len(val_subjects)} subjects)")
print(f"  Test:  {len(test_df)} samples ({len(test_subjects)} subjects)")

# ── Summary report ─────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
    f.write("EmPath Dataset Split Summary\n")
    f.write("="*40 + "\n")
    f.write(f"Total subjects: 50\n")
    f.write(f"Train: 35 subjects, {len(train_df)} samples\n")
    f.write(f"Val:   7 subjects,  {len(val_df)} samples\n")
    f.write(f"Test:  8 subjects,  {len(test_df)} samples\n")
    f.write(f"Classes: PA2 vs PA3\n")

print("Summary report saved ✓")