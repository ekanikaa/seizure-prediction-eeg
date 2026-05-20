# Seizure Prediction from EEG Signals
### Early-warning ML pipeline achieving pre-ictal detection ~131 seconds before seizure onset

---

## Overview

Epileptic seizures affect ~50 million people worldwide. This project builds a machine learning pipeline that predicts seizure onset from raw EEG signals, giving patients and caregivers a meaningful warning window before an episode occurs.

Using the **CHB-MIT Scalp EEG benchmark dataset**, the system classifies EEG windows as **ictal** (seizure) or **interictal** (normal), with a detection window of approximately **131 seconds pre-onset**.

---

## Pipeline Architecture

```
Raw EEG (.edf files)
       │
       ▼
  load_eeg.py         ← Load and segment EEG recordings
       │
       ▼
feature_engineering.py ← Extract time & frequency domain features
       │
       ▼
  risk_model.py       ← Train & evaluate ML classifiers
       │
       ▼
  controller.py       ← SDN-inspired orchestration layer
       │
       ▼
  dashboard.py        ← Real-time risk visualisation
```

---

## Models Compared

Three classifiers evaluated under **Stratified K-Fold cross-validation** with class-imbalance handling:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; interpretable coefficients |
| Random Forest | Ensemble; handles non-linear feature interactions |
| SVM (RBF kernel) | Strong performance on high-dimensional EEG features |

All models trained with `class_weight="balanced"` to handle the heavily skewed ictal/interictal ratio.

Evaluation metrics: **ROC-AUC, Confusion Matrix, Classification Report (Precision / Recall / F1)**

---

## Key Features Engineered

From raw EEG time-series windows:
- Statistical features: mean, variance, skewness, kurtosis
- Frequency domain: power spectral density across EEG bands (delta, theta, alpha, beta, gamma)
- Signal complexity measures

---

## Dataset

**CHB-MIT Scalp EEG Database** — publicly available via PhysioNet  
- Patient: chb01 (focal seizures)
- Format: `.edf` (European Data Format)
- Sampling rate: 256 Hz

> Raw `.edf` data files are not included in this repository due to size.  
> Download from: https://physionet.org/content/chbmit/1.0.0/

Place downloaded files in `data/chb01/` before running.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/ekanikaa/seizure-prediction-eeg.git
cd seizure-prediction-eeg
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full pipeline**
```bash
python src/run_pipeline.py
```

**4. Launch dashboard**
```bash
python src/dashboard.py
```

---

## Requirements

```
numpy
scikit-learn
mne
pandas
scipy
```

---

## Project Structure

```
seizure-prediction-eeg/
├── src/
│   ├── load_eeg.py              # EEG loading and windowing
│   ├── feature_engineering.py  # Feature extraction
│   ├── risk_model.py            # Model training and evaluation
│   ├── controller.py            # SDN orchestration layer
│   ├── run_pipeline.py          # End-to-end pipeline runner
│   └── dashboard.py             # Real-time risk visualisation
├── data/                        # EDF files (not tracked — see above)
├── .gitignore
└── README.md
```

---

## Academic Context

This project was developed as a **Final Year B.Tech Project** at SRM Institute of Science and Technology, Chennai, under the domain of Software Defined Networking applied to biomedical signal processing.

An SDN-inspired controller layer (`controller.py`) manages signal flow between pipeline components, enabling modular, reconfigurable architecture — analogous to how SDN decouples control and data planes in networks.

---

## Author

**Ekanika Shah**  
B.Tech Computer Science, SRMIST Chennai  
B.S. Data Science, IIT Madras  
github.com/ekanikaa
