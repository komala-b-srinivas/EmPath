#!/bin/bash
#SBATCH --job-name=empath_shap
#SBATCH --output=%j_shap.out
#SBATCH --error=%j_shap.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq

# ── Environment ────────────────────────────────────────────────────────────
module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v

PYTHON=/cm/shared/spack/opt/spack/linux-rhel8-x86_64/gcc-8.5.0/miniconda3-24.7.1-bxh7x2vu2tfweqp3b43qu55j3ksmdav3/bin/python

# ── Install required packages ────────────────────────────────────────────────
$PYTHON -m pip install --quiet shap matplotlib seaborn scikit-learn pandas numpy

# ── Run ────────────────────────────────────────────────────────────────────
cd ~/EmPath_v2
echo "=== SHAP Analysis starting at $(date) ==="
$PYTHON SRC/preprocessing/shap_analysis_loso.py
echo "=== Done at $(date) ==="
