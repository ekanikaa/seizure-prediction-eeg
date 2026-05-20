# Seizure Prediction from EEG Signals
Early-warning ML pipeline achieving pre-ictal detection ~131 seconds before seizure onset

## Overview
Epileptic seizures affect ~50 million people worldwide. This project builds a machine learning pipeline that predicts seizure onset from raw EEG signals, giving patients and caregivers a meaningful warning window before an episode occurs.

Using the CHB-MIT Scalp EEG benchmark dataset, the system classifies EEG windows as ictal (seizure) or interictal (normal), with a detection window of approximately 131 seconds pre-onset.

## Models
Three classifiers evaluated under Stratified K-Fold cross-validation:
- Logistic Regression
- Random Forest
- SVM (RBF kernel)

Evaluation metrics: ROC-AUC, Confusion Matrix, Precision/Recall/F1

## Dataset
CHB-MIT Scalp EEG Database — publicly available via PhysioNet
Download from: https://physionet.org/content/chbmit/1.0.0/
Place files in data/chb01/ before running.

## How to Run
1. pip install -r requirements.txt
2. python src/run_pipeline.py
3. python src/dashboard.py

## Author
Ekanika Shah — github.com/ekanikaa
