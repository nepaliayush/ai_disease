"""Bias / fairness evaluation of the deployed clinical models.

Evaluates each model's performance across the demographic subgroups that the
public datasets permit:
  - diabetes: age quartile groups        (no sex/ethnicity metadata available)
  - heart:    sex + age groups
  - liver:    sex + age groups
  - ckd:      age groups                 (no sex/ethnicity metadata available)

Reports per-subgroup performance (accuracy, precision, recall, specificity,
F1, ROC-AUC) plus two fairness gap metrics on a stratified 80:20 test split:
  - demographic parity gap  (difference in predicted-positive rates)
  - equal opportunity gap   (difference in recall across subgroups)

Ethnicity is not recorded in any of the four public datasets; this is documented
as a limitation. Only public, anonymized datasets are used.

Run: python -m training.evaluate_bias
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from app.fields import DATASET_FEATURES

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AGE_BINS = [0, 40, 60, np.inf]
AGE_LABELS = ["<40", "40-60", ">60"]

DATASETS = [
    ("diabetes", "diabetes.csv", "age", ["age"]),
    ("heart_disease", "heart.csv", "age", ["age", "sex"]),
    ("liver_disease", "liver.csv", "age", ["age", "sex"]),
    ("ckd", "ckd.csv", "age", ["age"]),
]

SEX_LABELS = {0: "female", 1: "male"}


def subgroup_metrics(y_true, y_pred, y_proba):
    n = len(y_true)
    if n == 0:
        return {}
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return {
        "n": int(n),
        "actual_positive_rate": round(float(y_true.mean()), 4),
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4) if n > 1 and len(set(y_true)) > 1 else None,
    }


def evaluate(disease: str, file: str, age_col: str, meta_cols: list[str]) -> dict:
    artifact = joblib.load(MODELS_DIR / f"clinical_{disease}.joblib")
    features = DATASET_FEATURES[disease]
    df = pd.read_csv(PROCESSED_DATA_DIR / file)
    X, y = df[features].copy(), df["outcome"].astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xte = artifact["preprocessor"].transform(X_test)
    y_pred = artifact["model"].predict(Xte)
    y_proba = artifact["model"].predict_proba(Xte)[:, 1]

    age_group = pd.cut(X_test[age_col], bins=AGE_BINS, labels=AGE_LABELS)
    groups: dict[str, pd.Series] = {}
    for label in AGE_LABELS:
        mask = (age_group == label).values
        groups[f"age_{label}"] = mask
    if "sex" in meta_cols:
        for code, label in SEX_LABELS.items():
            groups[f"sex_{label}"] = (X_test["sex"] == code).values

    report = {}
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        report[name] = subgroup_metrics(y_test.iloc[idx], y_pred[idx], y_proba[idx])

    # Fairness gap metrics (over the groups with n >= 10).
    valid = {k: v for k, v in report.items() if v and v["n"] >= 10}
    if valid:
        pprs = [v["predicted_positive_rate"] for v in valid.values()]
        recalls = [v["recall"] for v in valid.values()]
        report["_fairness_gaps"] = {
            "max_demographic_parity_gap": round(max(pprs) - min(pprs), 4),
            "max_equal_opportunity_gap": round(max(recalls) - min(recalls), 4),
            "groups_evaluated": list(valid.keys()),
        }
    report["_metadata_available"] = meta_cols
    report["_metadata_absent"] = [c for c in ["age", "sex", "ethnicity"] if c not in meta_cols]
    return report


def main() -> None:
    results = {}
    for disease, file, age_col, meta_cols in DATASETS:
        print(f"=== {disease} ===")
        report = evaluate(disease, file, age_col, meta_cols)
        results[disease] = report
        for k, v in report.items():
            if k.startswith("_"):
                print(f"  {k}: {v}")
            else:
                print(f"  {k:12s} n={v['n']:4d} acc={v['accuracy']:.3f} "
                      f"rec={v['recall']:.3f} spec={v['specificity']:.3f} "
                      f"ppr={v['predicted_positive_rate']:.3f} "
                      f"apr={v['actual_positive_rate']:.3f} auc={v['roc_auc']}")

    out = REPORTS_DIR / "bias_report.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved ->", out)


if __name__ == "__main__":
    sys.exit(main())