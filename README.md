# EmPath — Multimodal Pain Intensity Detection

Capstone Project | CSC 303 | Graduate Software Engineering

## Overview

EmPath is a multimodal pain classification system that distinguishes between PA2 and PA3 pain levels using physiological biosignals and facial landmark features from the BioVid Heat Pain Database.

## Live Demo

[EmPath Demo](https://empath.streamlit.app) ← replace with your actual URL

## Results

| Model | Evaluation | Accuracy |
|---|---|---|
| Biosignal RF only | LOSO 67 subjects | 63.1% |
| Landmark RF only | LOSO 67 subjects | 61.4% |
| EmPath Stacked Fusion | LOSO 67 subjects | 65.3% |

- AUC-ROC: 0.719
- F1 Score: 0.653

## System Architecture

- Level 1: Random Forest on 35 biosignal features
- Level 1: Random Forest on 22 facial landmark features
- Level 2: Logistic Regression meta-learner
- Evaluation: Leave-One-Subject-Out cross-validation on 67 reactive subjects

## Repository Structure