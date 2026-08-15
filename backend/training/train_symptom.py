"""Train the symptom triage multi-class classifier.

Trains a Random Forest over the symptom-disease dataset (42 classes after the
grounded CKD augmentation, 132 binary symptom features). Outputs a probability
distribution over disease classes; per-disease symptom relevance scores are
derived downstream (see app/symptoms.py).

Metrics reported: macro accuracy and macro F1 (and per-class F1).

Run: python -m training.train_symptom
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from app.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "symptoms.csv")
    features = [c for c in df.columns if c != "disease"]
    X = df[features].values.astype(float)
    y = df["disease"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                 min_samples_leaf=1, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    artifact = {
        "model": clf,
        "features": features,
        "classes": list(clf.classes_),
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        },
    }
    joblib.dump(artifact, MODELS_DIR / "symptom_triage.joblib")
    with open(REPORTS_DIR / "symptom_metrics.json", "w") as f:
        json.dump(artifact["metrics"], f, indent=2)
    print("Saved ->", MODELS_DIR / "symptom_triage.joblib")


if __name__ == "__main__":
    sys.exit(main())