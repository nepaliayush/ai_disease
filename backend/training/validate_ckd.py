"""Robustness + explainability validation of the deployed CKD model.

The CKD set is near-perfectly separable, so a flat 100% accuracy is a dataset
artifact, not a modelling win. This script does not retrain anything — it
validates the deployed artifact two ways:

(a) Robustness: re-scores a *noised* copy of the held-out test inputs. Gaussian
    noise is added at several magnitudes (as a fraction of each feature's
    standard deviation, applied after the fitted preprocessor) to see how much
    performance the deployed model gives up under measurement error.
(b) Sanity: computes SHAP values over the held-out test set and ranks features
    by mean |SHAP| (and sign), so it is easy to check whether the model leans
    on clinically plausible markers (specific gravity, hemoglobin, serum
    creatinine) rather than spurious ones.

Run: python -m training.validate_ckd
Output: reports/ckd_validation.json
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
from app.shap_engine import explain

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Noise magnitudes as a fraction of each feature's std (after preprocessing).
NOISE_SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0]
SHAP_MAX_ROWS = 150


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def noise_robustness(artifact: dict, Xte: np.ndarray, y_test: pd.Series,
                     rng: np.random.Generator) -> list[dict]:
    """Re-score the held-out test set with additive Gaussian noise on inputs."""
    threshold = artifact["threshold"]
    rows = []
    for frac in NOISE_SIGMAS:
        Xn = Xte.copy()
        if frac > 0:
            noise = rng.normal(0.0, frac * Xte.std(axis=0, ddof=1), size=Xte.shape)
            Xn = Xte + noise
        proba = artifact["model"].predict_proba(Xn)[:, 1]
        pred = (proba >= threshold).astype(int)
        rows.append({"noise_sigma_frac": frac, **_metrics(y_test, pred, proba)})
    return rows


def shap_sanity(artifact: dict, Xte: np.ndarray, y_test: pd.Series) -> dict:
    """Mean |SHAP| and mean SHAP per feature over the held-out test set."""
    features = artifact["features"]
    model = artifact["shap_model"]
    background = artifact["background"]
    is_tree = artifact["is_tree_based"]

    n = min(len(Xte), SHAP_MAX_ROWS)
    sums_abs = np.zeros(len(features))
    sums = np.zeros(len(features))
    for i in range(n):
        _ev, vals = explain(model, is_tree, Xte[i], background, class_index=1)
        vals = np.asarray(vals).reshape(-1)
        sums_abs += np.abs(vals)
        sums += vals
    top = sorted(
        range(len(features)),
        key=lambda j: sums_abs[j],
        reverse=True,
    )
    return {
        "rows_used": n,
        "features": [
            {
                "feature": features[j],
                "mean_abs_shap": round(float(sums_abs[j] / n), 4),
                "mean_shap": round(float(sums[j] / n), 4),
            }
            for j in top
        ],
    }


def main() -> None:
    artifact = joblib.load(MODELS_DIR / "clinical_ckd.joblib")
    features = DATASET_FEATURES["ckd"]
    df = pd.read_csv(PROCESSED_DATA_DIR / "ckd.csv")
    X, y = df[features].copy(), df["outcome"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    Xte = artifact["preprocessor"].transform(X_test)

    print(f"Deployed CKD model: {artifact['model_name']} "
          f"(threshold={artifact['threshold']:.2f}, "
          f"calibrated={artifact.get('calibrated', False)})")
    print(f"Baseline on held-out split: "
          f"{_metrics(y_test, artifact['model'].predict(Xte), artifact['model'].predict_proba(Xte)[:, 1])}")

    rng = np.random.default_rng(42)
    print("\n=== (a) Noise robustness (Gaussian noise on test inputs) ===")
    print(f"{'sigma_frac':>10} {'acc':>7} {'prec':>6} {'recall':>7} {'f1':>6} {'auc':>6}")
    rows = noise_robustness(artifact, Xte, y_test, rng)
    for r in rows:
        print(f"{r['noise_sigma_frac']:10.2f} {r['accuracy']:7.3f} {r['precision']:6.3f} "
              f"{r['recall']:7.3f} {r['f1']:6.3f} {r['roc_auc']:6.3f}")

    print("\n=== (b) SHAP sanity check (top features by mean |SHAP|) ===")
    shap = shap_sanity(artifact, Xte, y_test)
    print(f"(rows used: {shap['rows_used']})")
    print(f"{'feature':<24} {'mean_abs_shap':>14} {'mean_shap':>10}")
    for f in shap["features"][:15]:
        print(f"{f['feature']:<24} {f['mean_abs_shap']:14.4f} {f['mean_shap']:10.4f}")

    report = {
        "deployed_model": artifact["model_name"],
        "baseline": _metrics(y_test, artifact["model"].predict(Xte),
                             artifact["model"].predict_proba(Xte)[:, 1]),
        "noise_robustness": rows,
        "shap_top_features": shap["features"],
    }
    out = REPORTS_DIR / "ckd_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print("\nSaved ->", out)


if __name__ == "__main__":
    sys.exit(main())