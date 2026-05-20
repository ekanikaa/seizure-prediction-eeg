import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ────────────────────────────────────────────
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print(f"Dataset: {X.shape[0]} windows, {X.shape[1]} features")
print(f"Ictal: {sum(y)}  |  Interictal: {len(y) - sum(y)}\n")

# ── MODELS ───────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", class_weight="balanced",
        probability=True, random_state=42
    ),
}

# ── EVALUATION ───────────────────────────────────────────
RECORDING_HOURS = 3.0  # 3 files × 1 hour each

def evaluate(model, X, y):
    """
    Stratified 5-fold cross validation.
    Returns sensitivity, FPR/hr, AUC-ROC.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sensitivities, fprs, aucs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_hr      = fp / RECORDING_HOURS
        auc         = roc_auc_score(y_test, y_prob)

        sensitivities.append(sensitivity)
        fprs.append(fpr_hr)
        aucs.append(auc)

    return {
        "sensitivity": np.mean(sensitivities),
        "fpr_hr":      np.mean(fprs),
        "auc":         np.mean(aucs),
    }

# ── RUN ──────────────────────────────────────────────────
print("=" * 55)
print(f"{'Model':<25} {'Sensitivity':>12} {'FPR/hr':>8} {'AUC':>8}")
print("=" * 55)

results = {}
for name, model in models.items():
    r = evaluate(model, X, y)
    results[name] = r
    print(f"{name:<25} {r['sensitivity']*100:>11.1f}%"
          f" {r['fpr_hr']:>8.2f} {r['auc']:>8.3f}")

print("=" * 55)

# ── PICK WINNER ──────────────────────────────────────────
best = max(results, key=lambda m: results[m]["sensitivity"])
print(f"\n✅ Best model by sensitivity: {best}")
print(f"   Sensitivity : {results[best]['sensitivity']*100:.1f}%")
print(f"   FPR/hr      : {results[best]['fpr_hr']:.2f}")
print(f"   AUC-ROC     : {results[best]['auc']:.3f}")
print(f"\n→ {best} will be wired into the SDN controller.")