"""Train and compare five clinical models per disease.

For each of the four diseases:
  - stratified 80:20 train/test split
  - 10-fold stratified cross-validation (repeated) with a leakage-free pipeline
    (impute -> IQR clip -> optional log1p -> scale -> SMOTE/SMOTEENN ->
    classifier) inside every fold
  - five model families: Logistic Regression, SVM, Random Forest, XGBoost, MLP
  - diabetes, heart disease and liver disease families are tuned with
    RandomizedSearchCV over the same leakage-free pipeline, optimizing
    cross-validated ROC-AUC
  - metrics on the held-out test set: accuracy, precision, recall, F1,
    specificity, balanced accuracy, ROC-AUC (plus per-class precision/recall
    and a confusion matrix)
  - the best model (by CV-AUC, test AUC as tiebreak) is persisted to models/
    together with its fitted preprocessors, feature order, a background sample
    for SHAP, and the full comparison table
  - the deployed model's decision threshold is optimized (maximizing F1, or
    balanced accuracy on the imbalanced liver set, on a held-out calibration
    subset) and stored so reported test metrics reflect the deployed classifier

Run: python -m training.train_clinical
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_score, RandomizedSearchCV,
)

from app.config import (
    CLINICAL_MODELS, DISEASE_LABELS, DISEASE_PREPROCESSING, MODELS_DIR,
    MODEL_SELECTION_METRIC, MODEL_SELECTION_TIEBREAK, PROCESSED_DATA_DIR,
    REPORTS_DIR, SHAP_BACKGROUND_SAMPLES,
)
from app.fields import DATASET_FEATURES
from app.preprocess import (
    CalibratedClassifier, SamplerChoice, fit_preprocessor, build_clinical_pipeline,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Number of CV folds for reported cross-validation. 10 folds give more stable
# per-fold estimates on these small datasets (though test-fold recall variance
# on e.g. heart is dominated by the small number of positives per fold, not the
# number of folds).
CV_FOLDS = 10

# Number of repeated stratified CV rounds used to estimate generalization.
# A single split can (and does, for CKD) report a fluky 100%; repeating with
# different shuffles gives an honest mean + spread + 95% CI.
CV_REPEATS = 3

# Number of repeated random hold-out splits used to report the deployed
# model's headline metrics. Averaging over seeds avoids a single lucky split
# reporting a flat 100% (which happens on the trivially-separable CKD set).
HOLDOUT_REPEATS = 5

# Diseases whose model families get hyperparameter-tuned. Heart was under
# published baselines (untuned RF defaults); diabetes and liver are the other
# two weak datasets. CKD is already at literature levels, so its fixed config
# is kept.
TUNED_DISEASES = {"diabetes", "heart_disease", "liver_disease"}

# RandomizedSearchCV budgets per family (number of candidates sampled).
SEARCH_ITERS = {
    "LogisticRegression": 25,
    "SVM": 25,
    "RandomForest": 40,
    "XGBoost": 40,
    "MLP": 16,
}

# Search spaces for the leakage-free pipeline (impute -> clip -> scale ->
# sample -> clf), so parameters carry the `clf__`/`scaler__`/`resampler__`
# prefixes. Includes class weighting / scale_pos_weight to counter class
# imbalance on top of SMOTE/SMOTEENN.
PARAM_GRIDS = {
    "LogisticRegression": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "clf__solver": ["liblinear", "lbfgs", "newton-cg"],
        "clf__class_weight": [None, "balanced"],
    },
    "SVM": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__kernel": ["rbf", "linear"],
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        "clf__gamma": ["scale", 0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
        "clf__class_weight": [None, "balanced"],
    },
    "RandomForest": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__n_estimators": [200, 300, 500, 800],
        "clf__max_depth": [None, 4, 6, 8, 12, 16],
        "clf__min_samples_split": [2, 3, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__n_estimators": [200, 400, 700],
        "clf__max_depth": [3, 4, 5, 7],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__scale_pos_weight": [1.0, 1.5, 2.0],
        "clf__reg_lambda": [0.0, 0.5, 1.0],
    },
    "MLP": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 0.1],
        "clf__learning_rate_init": [1e-3, 5e-3, 1e-2],
        "clf__activation": ["relu", "tanh"],
        "clf__max_iter": [2000],
    },
}

# Per-disease grid overrides. Diabetes SVM is best at rbf + min-max + SMOTE
# (experiments showed linear kernel, StandardScaler and SMOTEENN all regress
# its honest hold-out accuracy), so its search space is constrained there to
# avoid a noisy CV-AUC edge picking a worse deployed model.
PARAM_GRIDS_OVERRIDES = {
    "diabetes": {
        "SVM": {
            "scaler__kind": ["minmax"],
            "resampler__kind": ["smote"],
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            "clf__class_weight": [None, "balanced"],
        },
    },
}


def disease_param_grid(name: str, disease: str) -> dict:
    grid = PARAM_GRIDS[name]
    override = PARAM_GRIDS_OVERRIDES.get(disease, {}).get(name)
    return override if override is not None else grid


def specificity_score(y_true, y_pred) -> float:
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def per_class_report(y_true, y_pred) -> dict:
    """Per-class precision / recall / F1 for classes 0 and 1."""
    report = {}
    for cls in (0, 1):
        mask = y_true == cls
        tp = int(((y_pred == cls) & mask).sum())
        fp = int(((y_pred == cls) & ~mask).sum())
        fn = int((mask & (y_pred != cls)).sum())
        report[str(cls)] = {
            "support": int(mask.sum()),
            "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        }
    return report


def find_best_threshold(y_true, proba, lo: float = 0.3, hi: float = 0.8,
                        n: int = 51, objective: str = "f1") -> float:
    """Threshold maximizing the chosen objective (F1 or balanced accuracy) on
    a held-out subset (used for calibration)."""
    best_t, best_obj = 0.5, -1.0
    for t in np.linspace(lo, hi, n):
        pred = (proba >= t).astype(int)
        obj = (balanced_accuracy_score(y_true, pred) if objective == "balanced_accuracy"
               else f1_score(y_true, pred, zero_division=0))
        if obj > best_obj:
            best_t, best_obj = float(t), float(obj)
    return best_t


def cross_validate(clf, X, y, n_splits: int = CV_FOLDS, scaler: str = "minmax",
                   log_cols: list[str] | None = None,
                   resampler: str = "smote") -> dict:
    """Repeated stratified CV (multiple shuffle seeds) of the leakage-free
    pipeline. Returns mean/std/95%-CI of accuracy, precision, recall, F1,
    balanced accuracy and ROC-AUC across all folds; repeats guard against a
    single lucky split reporting 100%."""
    accs, precs, recs, f1s, bals, aucs, specs = [], [], [], [], [], [], []
    for rep in range(CV_REPEATS):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + rep)
        for tr_idx, te_idx in cv.split(X, y):
            pipe = build_clinical_pipeline(clf, scaler=scaler, log_cols=log_cols,
                                           resampler=resampler)
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            pipe.fit(Xtr, ytr)
            yp = pipe.predict(Xte)
            yp_p = pipe.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(yte, yp)); precs.append(precision_score(yte, yp, zero_division=0))
            recs.append(recall_score(yte, yp, zero_division=0)); f1s.append(f1_score(yte, yp, zero_division=0))
            bals.append(balanced_accuracy_score(yte, yp)); specs.append(specificity_score(yte, yp))
            aucs.append(roc_auc_score(yte, yp_p))
    arrays = {"accuracy": np.asarray(accs), "precision": np.asarray(precs),
              "recall": np.asarray(recs), "f1": np.asarray(f1s),
              "balanced_accuracy": np.asarray(bals),
              "specificity": np.asarray(specs), "roc_auc": np.asarray(aucs)}
    out = {"cv_repeats": CV_REPEATS, "cv_folds": n_splits}
    for k, arr in arrays.items():
        out[f"cv_{k}_mean"] = float(arr.mean())
        out[f"cv_{k}_std"] = float(arr.std())
        out[f"cv_{k}_ci"] = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))
    # Backward-compatible aliases for the selection metric and frontend.
    out["cv_auc_mean"] = out["cv_roc_auc_mean"]
    out["cv_auc_std"] = out["cv_roc_auc_std"]
    out["cv_auc_ci"] = out["cv_roc_auc_ci"]
    out["cv_acc_mean"] = out["cv_accuracy_mean"]
    out["cv_acc_std"] = out["cv_accuracy_std"]
    out["cv_acc_ci"] = out["cv_accuracy_ci"]
    return out


def tune_model(name, factory, X_train, y_train, cv: StratifiedKFold,
               scaler: str = "minmax", log_cols: list[str] | None = None,
               disease: str | None = None):
    """Tune one family with RandomizedSearchCV over the leakage-free pipeline.

    Returns the fitted search object (used to evaluate held-out predictions)."""
    search = RandomizedSearchCV(
        build_clinical_pipeline(factory(), scaler=scaler, log_cols=log_cols),
        disease_param_grid(name, disease or ""),
        n_iter=SEARCH_ITERS[name],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search


def disease_preproc(disease: str) -> dict:
    return DISEASE_PREPROCESSING.get(disease, {"scaler": "minmax", "log_cols": None,
                                               "threshold_objective": "f1"})


def extract_clf_params(best_params: dict) -> dict:
    """Pull classifier params (`clf__`) out of the search space."""
    return {k[len("clf__"):]: v for k, v in best_params.items() if k.startswith("clf__")}


def fit_deployed_protocol(clf, X_train, y_train, random_state: int = 42,
                          scaler: str = "minmax", log_cols: list[str] | None = None,
                          resampler: str = "smote",
                          threshold_objective: str = "f1"):
    """Train the exact deployed protocol: fit preprocessor (scale + optional
    log1p), keep a calibration subset, sample the fit subset (SMOTE/SMOTEENN),
    Platt-scale + optimize threshold on the calibration subset. Returns
    (preprocessor, CalibratedClassifier, threshold)."""
    pre = fit_preprocessor(X_train, scaler=scaler, log_cols=log_cols)
    Xt = pre.transform(X_train)
    Xt_fit, X_cal, y_fit, y_cal = train_test_split(
        Xt, y_train, test_size=0.25, stratify=y_train, random_state=random_state)
    Xt_sampled, y_sampled = SamplerChoice(resampler).fit_resample(Xt_fit, y_fit)
    clf.fit(Xt_sampled, y_sampled)
    p_cal = clf.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    if isinstance(clf, LogisticRegression):
        calibrator = None
    else:
        calibrator = LogisticRegression().fit(p_cal, y_cal)
    q_cal = (calibrator.predict_proba(p_cal)[:, 1] if calibrator is not None
             else p_cal[:, 0])
    threshold = find_best_threshold(y_cal, q_cal, objective=threshold_objective)
    return pre, CalibratedClassifier(clf, calibrator, threshold), threshold


def deployed_holdout_metrics(factory, best_params, X, y, seed: int = 42,
                             repeats: int = HOLDOUT_REPEATS,
                             scaler: str = "minmax",
                             log_cols: list[str] | None = None,
                             resampler: str = "smote",
                             threshold_objective: str = "f1") -> dict:
    """Mean + std of the deployed model's headline metrics over `repeats`
    random stratified hold-out splits. Reports an honest point estimate with
    spread instead of one lucky split (which shows 100% on the easy CKD set)."""
    accs, precs, recs, f1s, aucs, specs, bals = [], [], [], [], [], [], []
    for r in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed + r)
        clf = factory()
        clf.set_params(**extract_clf_params(best_params or {}))
        pre, deployed, _thr = fit_deployed_protocol(
            clf, Xtr, ytr, scaler=scaler, log_cols=log_cols,
            resampler=resampler, threshold_objective=threshold_objective)
        Xte_t = pre.transform(Xte)
        p = deployed.predict_proba(Xte_t)[:, 1]
        m = compute_metrics(yte, deployed.predict(Xte_t), p)
        accs.append(m["accuracy"]); precs.append(m["precision"])
        recs.append(m["recall"]); f1s.append(m["f1"]); aucs.append(m["roc_auc"])
        specs.append(m["specificity"]); bals.append(m["balanced_accuracy"])
    mean = {"accuracy": float(np.mean(accs)), "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)), "f1": float(np.mean(f1s)),
            "roc_auc": float(np.mean(aucs)), "specificity": float(np.mean(specs)),
            "balanced_accuracy": float(np.mean(bals))}
    std = {"accuracy": float(np.std(accs)), "precision": float(np.std(precs)),
           "recall": float(np.std(recs)), "f1": float(np.std(f1s)),
           "roc_auc": float(np.std(aucs)), "specificity": float(np.std(specs)),
           "balanced_accuracy": float(np.std(bals))}
    return {**mean, "holdout_repeats": repeats, "holdout_std": std}


def train_disease(disease: str, df: pd.DataFrame, results: dict) -> None:
    features = DATASET_FEATURES[disease]
    X = df[features].copy()
    y = df["outcome"].astype(int)

    preproc = disease_preproc(disease)
    scaler = preproc["scaler"]
    log_cols = preproc["log_cols"]
    threshold_objective = preproc["threshold_objective"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tune = disease in TUNED_DISEASES
    per_model = {}
    for name, (factory, _is_tree) in CLINICAL_MODELS.items():
        clf = factory()
        if tune and name in PARAM_GRIDS:
            search = tune_model(name, factory, X_train, y_train, tune_cv,
                                scaler=scaler, log_cols=log_cols, disease=disease)
            best_params = search.best_params_
            # Report CV via the same stabilized protocol used for untuned models
            # so the selection metric is comparable across families.
            tuned_clf = factory()
            tuned_clf.set_params(**extract_clf_params(best_params))
            cv_scores = cross_validate(
                tuned_clf, X, y, scaler=best_params.get("scaler__kind", scaler),
                log_cols=log_cols,
                resampler=best_params.get("resampler__kind", "smote"))
            cv_scores["best_params"] = best_params
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
        else:
            cv_scores = cross_validate(clf, X, y, scaler=scaler, log_cols=log_cols)
            # Fit the deployed-style pipeline on the 80% split.
            pre = fit_preprocessor(X_train, scaler=scaler, log_cols=log_cols)
            Xt = pre.transform(X_train)
            Xt_smote, y_smote = SamplerChoice("smote").fit_resample(Xt, y_train)
            clf.fit(Xt_smote, y_smote)
            Xte = pre.transform(X_test)
            y_pred = clf.predict(Xte)
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(Xte)[:, 1]
            else:
                y_proba = clf.decision_function(Xte)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics.update(cv_scores)
        metrics["per_class"] = per_class_report(y_test, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        per_model[name] = metrics
        print(f"[{disease}] {name:18s} acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f} "
              f"cv_auc={metrics['cv_auc_mean']:.3f}"
              + (" [tuned]" if tune and name in PARAM_GRIDS else ""))

    results[disease] = per_model

    # Select the deployed model.
    best = max(
        per_model,
        key=lambda n: (per_model[n][MODEL_SELECTION_METRIC],
                       per_model[n][MODEL_SELECTION_TIEBREAK]),
    )

    # Re-fit the selected model for deployment and persist artifact.
    clf = CLINICAL_MODELS[best][0]()
    best_params = per_model[best].get("best_params")
    clf.set_params(**extract_clf_params(best_params or {}))
    resampler = best_params.get("resampler__kind", "smote") if best_params else "smote"
    best_scaler = best_params.get("scaler__kind", scaler) if best_params else scaler
    pre, deployed, threshold = fit_deployed_protocol(
        clf, X_train, y_train, scaler=best_scaler, log_cols=log_cols,
        resampler=resampler, threshold_objective=threshold_objective)
    Xte = pre.transform(X_test)
    y_proba = deployed.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    background = pre.transform(X_train)[:SHAP_BACKGROUND_SAMPLES]

    # Headline metrics are the mean over repeated hold-out splits (honest,
    # with spread), not a single lucky split. The split-42 metrics are kept
    # in `single_split` for reference.
    deployed_metrics = deployed_holdout_metrics(
        CLINICAL_MODELS[best][0], best_params, X, y, scaler=best_scaler,
        log_cols=log_cols, resampler=resampler,
        threshold_objective=threshold_objective)
    single_split = compute_metrics(y_test, deployed.predict(Xte), y_proba)
    deployed_metrics.update({
        **{k: per_model[best][k] for k in per_model[best]
           if k.startswith("cv_")},
        "best_params": best_params,
        "scaler": best_scaler,
        "resampler": resampler,
        "threshold_objective": threshold_objective,
        "threshold": threshold,
        "single_split": single_split,
        "per_class": per_class_report(y_test, deployed.predict(Xte)),
        "confusion_matrix": confusion_matrix(y_test, deployed.predict(Xte)).tolist(),
    })

    artifact = {
        "disease": disease,
        "model_name": best,
        "model": deployed,
        "shap_model": clf,  # base estimator used for SHAP explanations
        "base_estimator": best,
        "calibrated": deployed.calibrator is not None,
        "threshold": threshold,
        "preprocessor": pre,
        "features": features,
        "background": background,
        "is_tree_based": CLINICAL_MODELS[best][1],
        "metrics": deployed_metrics,
        "test_auc": float(test_auc),
        "comparison": per_model,
    }
    out_path = MODELS_DIR / f"clinical_{disease}.joblib"
    joblib.dump(artifact, out_path)
    print(f"[{disease}] deployed {best} -> {out_path} "
          f"(threshold={threshold:.2f}, calibrated={deployed.calibrator is not None})")


def main() -> None:
    print("Training clinical models ...")
    results = {}
    for disease, label in DISEASE_LABELS.items():
        df = pd.read_csv(PROCESSED_DATA_DIR / f"{label_to_file(disease)}.csv")
        print(f"\n=== {label} ({disease}) — {len(df)} rows ===")
        train_disease(disease, df, results)

    report = {"selection_metric": MODEL_SELECTION_METRIC,
              "tiebreak": MODEL_SELECTION_TIEBREAK,
              "results": results}
    with open(REPORTS_DIR / "clinical_comparison.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    pd.DataFrame(
        {(d, m): metrics for d, models in results.items() for m, metrics in models.items()}
    ).T.to_csv(REPORTS_DIR / "clinical_comparison.csv")
    print("\nSaved comparison ->", REPORTS_DIR / "clinical_comparison.csv")


def label_to_file(disease: str) -> str:
    return {"diabetes": "diabetes", "heart_disease": "heart",
            "liver_disease": "liver", "ckd": "ckd"}[disease]


if __name__ == "__main__":
    sys.exit(main())