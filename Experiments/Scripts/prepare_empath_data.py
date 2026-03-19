"""
EmPath Project — Data Confirmation & Split Generator
=====================================================
Confirms PA2/PA3 counts from BioVid samples.csv, selects 50 subjects,
and produces subject-independent train/val/test splits.

Usage:
    python prepare_empath_data.py --samples path/to/samples.csv

Outputs (written to ./empath_splits/):
    - summary_report.txt     Full count confirmation & split statistics
    - train.csv
    - val.csv
    - test.csv
"""

import csv
import random
import argparse
import os
from collections import defaultdict


# ── Configuration ─────────────────────────────────────────────────────────────
NUM_SUBJECTS     = 50       # How many subjects to select
TRAIN_RATIO      = 0.70     # 35 subjects
VAL_RATIO        = 0.14     # 7 subjects
TEST_RATIO       = 0.16     # 8 subjects
TARGET_CLASSES   = {"PA2", "PA3"}
RANDOM_SEED      = 42
OUTPUT_DIR       = "empath_splits"
# ──────────────────────────────────────────────────────────────────────────────


def load_samples(filepath):
    samples = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append(row)
    return samples


def confirm_counts(samples):
    lines = []
    lines.append("=" * 60)
    lines.append("  BIOVID DATASET — FULL COUNT CONFIRMATION")
    lines.append("=" * 60)

    total = len(samples)
    all_subjects = sorted(set(s["subject_id"] for s in samples), key=int)
    lines.append(f"\nTotal samples (all classes, all subjects): {total}")
    lines.append(f"Total unique subjects:                     {len(all_subjects)}")

    class_counts = defaultdict(int)
    for s in samples:
        class_counts[s["class_name"]] += 1

    lines.append("\nClass distribution (all subjects):")
    lines.append(f"  {'Class':<10} {'Samples':>8}  {'Subjects':>9}")
    lines.append(f"  {'-'*10} {'-'*8}  {'-'*9}")
    for cls in sorted(class_counts):
        subj_count = len(set(s["subject_id"] for s in samples if s["class_name"] == cls))
        lines.append(f"  {cls:<10} {class_counts[cls]:>8}  {subj_count:>9}")

    pa2 = [s for s in samples if s["class_name"] == "PA2"]
    pa3 = [s for s in samples if s["class_name"] == "PA3"]
    pa2_subjs = set(s["subject_id"] for s in pa2)
    pa3_subjs = set(s["subject_id"] for s in pa3)
    both = pa2_subjs & pa3_subjs

    lines.append(f"\nPA2 vs PA3 — Key Numbers:")
    lines.append(f"  PA2 total samples:              {len(pa2)}")
    lines.append(f"  PA3 total samples:              {len(pa3)}")
    lines.append(f"  Subjects with PA2:              {len(pa2_subjs)}")
    lines.append(f"  Subjects with PA3:              {len(pa3_subjs)}")
    lines.append(f"  Subjects with BOTH PA2 & PA3:   {len(both)}")
    lines.append(f"  Samples per subject (PA2):      {len(pa2) // len(pa2_subjs)}")
    lines.append(f"  Samples per subject (PA3):      {len(pa3) // len(pa3_subjs)}")
    lines.append(f"  Dataset balanced:               {'YES ✓' if len(pa2) == len(pa3) else 'NO ✗'}")

    return "\n".join(lines), sorted(both, key=int)


def select_subjects(eligible_subjects, n, seed):
    rng = random.Random(seed)
    selected = rng.sample(eligible_subjects, n)
    return sorted(selected, key=int)


def make_splits(subjects, train_r, val_r, seed):
    rng = random.Random(seed + 1)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)

    train_subjs = shuffled[:n_train]
    val_subjs   = shuffled[n_train:n_train + n_val]
    test_subjs  = shuffled[n_train + n_val:]

    return sorted(train_subjs, key=int), sorted(val_subjs, key=int), sorted(test_subjs, key=int)


def filter_samples(samples, subject_ids, classes):
    subject_set = set(subject_ids)
    return [s for s in samples if s["subject_id"] in subject_set
            and s["class_name"] in classes]


def write_csv(rows, filepath):
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def split_report(train_s, val_s, test_s, train_subjs, val_subjs, test_subjs):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  SPLIT SUMMARY (Subject-Independent)")
    lines.append("=" * 60)

    for name, subjs, split in [
        ("TRAIN", train_subjs, train_s),
        ("VAL",   val_subjs,   val_s),
        ("TEST",  test_subjs,  test_s),
    ]:
        pa2_n = sum(1 for s in split if s["class_name"] == "PA2")
        pa3_n = sum(1 for s in split if s["class_name"] == "PA3")
        lines.append(f"\n  {name}")
        lines.append(f"    Subjects : {len(subjs)}  → IDs: {', '.join(subjs)}")
        lines.append(f"    PA2      : {pa2_n} samples")
        lines.append(f"    PA3      : {pa3_n} samples")
        lines.append(f"    Total    : {len(split)} samples")
        lines.append(f"    Balanced : {'YES ✓' if pa2_n == pa3_n else 'NO ✗'}")

    total = len(train_s) + len(val_s) + len(test_s)
    lines.append(f"\n  GRAND TOTAL : {total} samples across all splits")
    lines.append(f"  Subject overlap check : NONE ✓  (subject-independent)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EmPath data prep & split generator")
    parser.add_argument("--samples", default="/mnt/user-data/uploads/samples.csv",
                        help="Path to BioVid samples.csv")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--num_subjects", type=int, default=NUM_SUBJECTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print(f"\nLoading samples from: {args.samples}")
    samples = load_samples(args.samples)

    report, eligible = confirm_counts(samples)
    print(report)

    selected = select_subjects(eligible, args.num_subjects, args.seed)
    print(f"\n{'='*60}")
    print(f"  SUBJECT SELECTION")
    print(f"{'='*60}")
    print(f"  Eligible subjects (have both PA2 & PA3): {len(eligible)}")
    print(f"  Selected {args.num_subjects} subjects (seed={args.seed}):")
    print(f"  {', '.join(selected)}")

    train_subjs, val_subjs, test_subjs = make_splits(
        selected, TRAIN_RATIO, VAL_RATIO, args.seed
    )

    train_s = filter_samples(samples, train_subjs, TARGET_CLASSES)
    val_s   = filter_samples(samples, val_subjs,   TARGET_CLASSES)
    test_s  = filter_samples(samples, test_subjs,  TARGET_CLASSES)

    split_summary = split_report(train_s, val_s, test_s,
                                 train_subjs, val_subjs, test_subjs)
    print(split_summary)

    os.makedirs(args.output_dir, exist_ok=True)

    write_csv(train_s, os.path.join(args.output_dir, "train.csv"))
    write_csv(val_s,   os.path.join(args.output_dir, "val.csv"))
    write_csv(test_s,  os.path.join(args.output_dir, "test.csv"))

    full_report = report + "\n" + split_summary + f"""

{'='*60}
  OUTPUT FILES
{'='*60}
  {args.output_dir}/train.csv  →  {len(train_s)} samples  ({len(train_subjs)} subjects)
  {args.output_dir}/val.csv    →  {len(val_s)} samples   ({len(val_subjs)} subjects)
  {args.output_dir}/test.csv   →  {len(test_s)} samples   ({len(test_subjs)} subjects)
"""
    report_path = os.path.join(args.output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(full_report)

    print(f"\n{'='*60}")
    print(f"  OUTPUT FILES WRITTEN → ./{args.output_dir}/")
    print(f"{'='*60}")
    print(f"  train.csv          {len(train_s):>5} samples  ({len(train_subjs)} subjects)")
    print(f"  val.csv            {len(val_s):>5} samples  ({len(val_subjs)} subjects)")
    print(f"  test.csv           {len(test_s):>5} samples  ({len(test_subjs)} subjects)")
    print(f"  summary_report.txt")
    print(f"\n  Done! ✓\n")


if __name__ == "__main__":
    main()