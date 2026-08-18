"""Sweep decision thresholds of a deployed clinical model and report the
precision / recall (sensitivity) / specificity / F1 trade-off at each, so a
threshold can be chosen for a given operating point (e.g. high recall for
screening).

The sweep evaluates the deployed artifact (preprocessor + calibrated model) on
the same stratified 80:20 hold-out split used at training time (seed 42), then
re-derives hard predictions at every threshold from the model's probabilities.

Usage:
    python -m training.sweep_thresholds                      # heart_disease
    python -m training.sweep_thresholds --disease diabetes
    python -m training.sweep_thresholds --lo 0.1 --hi 0.9 --step 0.05

Output:
    - console table of threshold -> precision/recall/specificity/F1 + #pred
    - reports/threshold_sweep_<disease>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from app.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from app.fields import DATASET_FEATURES

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_FILE = {
    "diabetes": "diabetes.csv",
    "heart_disease": "heart.csv",
    "liver_disease": "liver.csv",
    "ckd": "ckd.csv",
}


def sweep(disease: str, lo: float, hi: float, step: float) -> pd.DataFrame:
    artifact = joblib.load(MODELS_DIR / f"clinical_{disease}.joblib")
    features = artifact["features"]
    df = pd.read_csv(PROCESSED_DATA_DIR / DISEASE_FILE[disease])
    X, y = df[features].copy(), df["outcome"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    Xte = artifact["preprocessor"].transform(X_test)
    proba = artifact["model"].predict_proba(Xte)[:, 1]

    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "specificity": round(float(
                ((y_test == 0) & (pred == 0)).sum() / max((y_test == 0).sum(), 1)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "n_predicted_positive": int(pred.sum()),
            "n_test": int(len(y_test)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disease", default="heart_disease",
                    choices=list(DISEASE_FILE), help="Deployed clinical model to sweep.")
    ap.add_argument("--lo", type=float, default=0.10, help="Lowest threshold (default 0.10)")
    ap.add_argument("--hi", type=float, default=0.90, help="Highest threshold (default 0.90)")
    ap.add_argument("--step", type=float, default=0.05, help="Threshold step (default 0.05)")
    args = ap.parse_args()

    art = joblib.load(MODELS_DIR / f"clinical_{args.disease}.joblib")
    table = sweep(args.disease, args.lo, args.hi, args.step)

    print(f"Deployed model: {art['model_name']} "
          f"(threshold={art.get('threshold', 0.5):.2f}, "
          f"calibrated={art.get('calibrated', False)})")
    print(f"Threshold sweep on the held-out split ({args.disease})")
    print(f"{'thr':>5} {'acc':>6} {'prec':>6} {'recall':>7} {'spec':>6} {'f1':>6} {'pred+/n':>9}")
    for r in table.itertuples():
        print(f"{r.threshold:5.2f} {r.accuracy:6.3f} {r.precision:6.3f} "
              f"{r.recall:7.3f} {r.specificity:6.3f} {r.f1:6.3f} "
              f"{r.n_predicted_positive}/{r.n_test}")

    # Helpers for picking a screening-oriented threshold.
    hi_recall = table[table["recall"] >= 0.90]
    if not hi_recall.empty:
        best = hi_recall.sort_values("f1", ascending=False).iloc[0]
        print(f"\nHighest-F1 threshold with recall >= 0.90: "
              f"{best['threshold']:.2f} "
              f"(recall={best['recall']:.3f}, precision={best['precision']:.3f}, "
              f"f1={best['f1']:.3f})")
    best_f1 = table.sort_values("f1", ascending=False).iloc[0]
    print(f"Max-F1 threshold: {best_f1['threshold']:.2f} "
          f"(recall={best_f1['recall']:.3f}, precision={best_f1['precision']:.3f}, "
          f"f1={best_f1['f1']:.3f})")

    out = REPORTS_DIR / f"threshold_sweep_{args.disease}.csv"
    table.to_csv(out, index=False)
    print("Saved ->", out)


if __name__ == "__main__":
    sys.exit(main())