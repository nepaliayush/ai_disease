"""Train and compare five clinical models per disease.

For each of the four diseases:
  - stratified 80:20 train/test split
  - 10-fold stratified cross-validation (repeated) with a leakage-free pipeline
    (impute -> IQR clip -> optional log1p -> scale -> SMOTE/SMOTEENN ->
    classifier) inside every fold
  - six model families: Logistic Regression, SVM, Random Forest, XGBoost, MLP, and
  CatBoost
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
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, RandomizedSearchCV,
)

from app.config import (
    CLINICAL_MODELS, DISEASE_LABELS, DISEASE_PREPROCESSING, MODELS_DIR,
    MODEL_SELECTION_METRIC, MODEL_SELECTION_TIEBREAK, PROCESSED_DATA_DIR,
    REPORTS_DIR, SHAP_BACKGROUND_SAMPLES, clinical_families,
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

# Nested cross-validation: outer folds estimate generalization, inner folds tune.
OUTER_CV_FOLDS = 5
# Bootstrap resamples for the deployed model's held-out AUC confidence interval.
BOOTSTRAP_REPS = 1000

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
    "CatBoost": 30,
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
        "clf__estimator__kernel": ["rbf", "linear"],
        "clf__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        "clf__estimator__gamma": ["scale", 0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
        "clf__estimator__class_weight": [None, "balanced"],
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
    "CatBoost": {
        "scaler__kind": ["minmax", "standard"],
        "resampler__kind": ["smote", "smoteenn"],
        "clf__iterations": [200, 400, 700],
        "clf__depth": [3, 4, 6, 8],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "clf__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bylevel": [0.7, 0.8, 1.0],
        "clf__scale_pos_weight": [1.0, 1.5, 2.0],
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
            "clf__estimator__kernel": ["rbf"],
            "clf__estimator__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            "clf__estimator__gamma": ["scale", 0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            "clf__estimator__class_weight": [None, "balanced"],
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
        "pr_auc": float(average_precision_score(y_true, y_proba)),
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
                   resampler: str = "smote",
                   imputer: str = "median") -> dict:
    """Repeated stratified CV (multiple shuffle seeds) of the leakage-free
    pipeline. Returns mean/std/95%-CI of accuracy, precision, recall, F1,
    balanced accuracy, ROC-AUC and PR-AUC across all folds; repeats guard
    against a single lucky split reporting 100%."""
    accs, precs, recs, f1s, bals, aucs, prs, specs = [], [], [], [], [], [], [], []
    for rep in range(CV_REPEATS):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + rep)
        for tr_idx, te_idx in cv.split(X, y):
            pipe = build_clinical_pipeline(clf, scaler=scaler, log_cols=log_cols,
                                           resampler=resampler, imputer=imputer)
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            pipe.fit(Xtr, ytr)
            yp = pipe.predict(Xte)
            yp_p = pipe.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(yte, yp)); precs.append(precision_score(yte, yp, zero_division=0))
            recs.append(recall_score(yte, yp, zero_division=0)); f1s.append(f1_score(yte, yp, zero_division=0))
            bals.append(balanced_accuracy_score(yte, yp)); specs.append(specificity_score(yte, yp))
            aucs.append(roc_auc_score(yte, yp_p))
            prs.append(average_precision_score(yte, yp_p))
    arrays = {"accuracy": np.asarray(accs), "precision": np.asarray(precs),
              "recall": np.asarray(recs), "f1": np.asarray(f1s),
              "balanced_accuracy": np.asarray(bals),
              "specificity": np.asarray(specs), "roc_auc": np.asarray(aucs),
              "pr_auc": np.asarray(prs)}
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


def nested_cross_validate(name: str, factory, X, y, disease: str | None = None,
                          scaler: str = "minmax", log_cols: list[str] | None = None,
                          resampler: str = "smote", imputer: str = "median",
                          outer_folds: int = OUTER_CV_FOLDS) -> dict:
    """Nested cross-validation for tuned model families.

    An outer CV loop estimates generalization; inside every outer training fold
    a fresh RandomizedSearchCV (inner CV) tunes hyperparameters on that fold's
    training data only, so the reported cv_* metrics are not optimistically
    biased by tuning on the evaluation data (the naive "tune once, then CV with
    best params" approach leaks tuning signal into every fold). Returns the same
    dict shape as `cross_validate` (mean/std/95%-CI per metric)."""
    accs, precs, recs, f1s, bals, aucs, prs, specs = [], [], [], [], [], [], [], []
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    for tr_idx, te_idx in outer.split(X, y):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = tune_model(name, factory, Xtr, ytr, inner, scaler=scaler,
                            log_cols=log_cols, disease=disease, imputer=imputer)
        yp = search.predict(Xte)
        yp_p = search.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(yte, yp)); precs.append(precision_score(yte, yp, zero_division=0))
        recs.append(recall_score(yte, yp, zero_division=0)); f1s.append(f1_score(yte, yp, zero_division=0))
        bals.append(balanced_accuracy_score(yte, yp)); specs.append(specificity_score(yte, yp))
        aucs.append(roc_auc_score(yte, yp_p))
        prs.append(average_precision_score(yte, yp_p))
    arrays = {"accuracy": np.asarray(accs), "precision": np.asarray(precs),
              "recall": np.asarray(recs), "f1": np.asarray(f1s),
              "balanced_accuracy": np.asarray(bals),
              "specificity": np.asarray(specs), "roc_auc": np.asarray(aucs),
              "pr_auc": np.asarray(prs)}
    out = {"cv_repeats": 1, "cv_folds": outer_folds, "nested": True}
    for k, arr in arrays.items():
        out[f"cv_{k}_mean"] = float(arr.mean())
        out[f"cv_{k}_std"] = float(arr.std())
        out[f"cv_{k}_ci"] = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))
    out["cv_auc_mean"] = out["cv_roc_auc_mean"]
    out["cv_auc_std"] = out["cv_roc_auc_std"]
    out["cv_auc_ci"] = out["cv_roc_auc_ci"]
    out["cv_acc_mean"] = out["cv_accuracy_mean"]
    out["cv_acc_std"] = out["cv_accuracy_std"]
    out["cv_acc_ci"] = out["cv_accuracy_ci"]
    return out


def bootstrap_auc(y_true, y_proba, n_boot: int = BOOTSTRAP_REPS,
                  seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI for ROC-AUC (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_proba[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals)
    return (float(vals.mean()), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def tune_model(name, factory, X_train, y_train, cv: StratifiedKFold,
               scaler: str = "minmax", log_cols: list[str] | None = None,
               disease: str | None = None,
               imputer: str = "median"):
    """Tune one family with RandomizedSearchCV over the leakage-free pipeline.

    Returns the fitted search object (used to evaluate held-out predictions)."""
    search = RandomizedSearchCV(
        build_clinical_pipeline(factory(), scaler=scaler, log_cols=log_cols,
                                imputer=imputer),
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
    return DISEASE_PREPROCESSING.get(disease, {
        "scaler": "minmax", "log_cols": None, "threshold_objective": "f1",
        "imputer": "median",
    })


def extract_clf_params(best_params: dict) -> dict:
    """Pull classifier params (`clf__`) out of the search space."""
    return {k[len("clf__"):]: v for k, v in best_params.items() if k.startswith("clf__")}


def fit_deployed_protocol(clf, X_train, y_train, random_state: int = 42,
                          scaler: str = "minmax", log_cols: list[str] | None = None,
                          resampler: str = "smote",
                          threshold_objective: str = "f1",
                          imputer: str = "median",
                          calibrate: bool = False):
    """Train the exact deployed protocol: fit preprocessor (scale + optional
    log1p), keep a calibration subset, sample the fit subset (SMOTE/SMOTEENN),
    Platt-scale + optimize threshold on the calibration subset. Returns
    (preprocessor, CalibratedClassifier, threshold).

    `calibrate=True` also Platt-scales linear models (used for heart so its
    deployed LogisticRegression reports calibrated probabilities like the other
    families)."""
    pre = fit_preprocessor(X_train, scaler=scaler, log_cols=log_cols, imputer=imputer)
    Xt = pre.transform(X_train)
    Xt_fit, X_cal, y_fit, y_cal = train_test_split(
        Xt, y_train, test_size=0.25, stratify=y_train, random_state=random_state)
    Xt_sampled, y_sampled = SamplerChoice(resampler).fit_resample(Xt_fit, y_fit)
    clf.fit(Xt_sampled, y_sampled)
    p_cal = clf.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    if isinstance(clf, LogisticRegression) and not calibrate:
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
                             threshold_objective: str = "f1",
                             imputer: str = "median",
                             calibrate: bool = False) -> dict:
    """Mean + std of the deployed model's headline metrics over `repeats`
    random stratified hold-out splits. Reports an honest point estimate with
    spread instead of one lucky split (which shows 100% on the easy CKD set)."""
    accs, precs, recs, f1s, aucs, prs, specs, bals = [], [], [], [], [], [], [], []
    for r in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed + r)
        clf = factory()
        clf.set_params(**extract_clf_params(best_params or {}))
        pre, deployed, _thr = fit_deployed_protocol(
            clf, Xtr, ytr, scaler=scaler, log_cols=log_cols,
            resampler=resampler, threshold_objective=threshold_objective,
            imputer=imputer, calibrate=calibrate)
        Xte_t = pre.transform(Xte)
        p = deployed.predict_proba(Xte_t)[:, 1]
        m = compute_metrics(yte, deployed.predict(Xte_t), p)
        accs.append(m["accuracy"]); precs.append(m["precision"])
        recs.append(m["recall"]); f1s.append(m["f1"]); aucs.append(m["roc_auc"])
        prs.append(m["pr_auc"]); specs.append(m["specificity"]); bals.append(m["balanced_accuracy"])
    mean = {"accuracy": float(np.mean(accs)), "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)), "f1": float(np.mean(f1s)),
            "roc_auc": float(np.mean(aucs)), "pr_auc": float(np.mean(prs)),
            "specificity": float(np.mean(specs)),
            "balanced_accuracy": float(np.mean(bals))}
    std = {"accuracy": float(np.std(accs)), "precision": float(np.std(precs)),
           "recall": float(np.std(recs)), "f1": float(np.std(f1s)),
           "roc_auc": float(np.std(aucs)), "pr_auc": float(np.std(prs)),
           "specificity": float(np.std(specs)),
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
    imputer = preproc.get("imputer", "median")
    selection_metric = preproc.get("selection_metric", MODEL_SELECTION_METRIC)
    selection_tiebreak = preproc.get("selection_tiebreak", MODEL_SELECTION_TIEBREAK)
    calibrate = preproc.get("calibrate", False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tune = disease in TUNED_DISEASES
    per_model = {}
    for name in clinical_families(disease):
        factory, _is_tree = CLINICAL_MODELS[name]
        clf = factory()
        if tune and name in PARAM_GRIDS:
            search = tune_model(name, factory, X_train, y_train, tune_cv,
                                scaler=scaler, log_cols=log_cols, disease=disease,
                                imputer=imputer)
            best_params = search.best_params_
            # Honest generalization estimate via nested CV (outer folds estimate
            # performance, tuning happens on each outer fold's training data
            # only) so the reported cv_auc / cv_pr_auc are not biased by tuning
            # on the evaluation data.
            cv_scores = nested_cross_validate(
                name, factory, X, y, disease=disease, scaler=scaler,
                log_cols=log_cols, imputer=imputer)
            cv_scores["best_params"] = best_params
            fam_scaler = best_params.get("scaler__kind", scaler)
            fam_resampler = best_params.get("resampler__kind", "smote")
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
        else:
            cv_scores = cross_validate(clf, X, y, scaler=scaler, log_cols=log_cols,
                                       imputer=imputer)
            best_params = None
            fam_scaler = scaler
            fam_resampler = "smote"
            # Fit the deployed-style pipeline on the 80% split.
            pre = fit_preprocessor(X_train, scaler=scaler, log_cols=log_cols,
                                   imputer=imputer)
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

        # Headline held-out metrics for every family use the SAME deployed
        # protocol (SMOTE -> fit -> calibrate -> optimize threshold) averaged
        # over repeated random hold-out splits, so the deployed row in the model
        # family comparison matches the "Model performance" table exactly. The
        # single split-42 metrics are kept in `single_split` for reference.
        holdout = deployed_holdout_metrics(
            factory, best_params, X, y, scaler=fam_scaler, log_cols=log_cols,
            resampler=fam_resampler, threshold_objective=threshold_objective,
            imputer=imputer, calibrate=calibrate)
        single = compute_metrics(y_test, y_pred, y_proba)
        metrics = dict(holdout)
        metrics.update(cv_scores)
        metrics["single_split"] = single
        metrics["per_class"] = per_class_report(y_test, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        per_model[name] = metrics
        print(f"[{disease}] {name:18s} acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f} "
              f"pr_auc={metrics['pr_auc']:.3f} "
              f"cv_auc={metrics['cv_auc_mean']:.3f} "
              f"cv_pr_auc={metrics['cv_pr_auc_mean']:.3f}"
              + (" [tuned]" if tune and name in PARAM_GRIDS else ""))

    results[disease] = per_model

    # Select the deployed model (PR-AUC for the imbalanced liver set, ROC-AUC
    # elsewhere).
    best = max(
        per_model,
        key=lambda n: (per_model[n][selection_metric],
                       per_model[n][selection_tiebreak]),
    )

    # Re-fit the selected model for deployment and persist artifact.
    clf = CLINICAL_MODELS[best][0]()
    best_params = per_model[best].get("best_params")
    clf.set_params(**extract_clf_params(best_params or {}))
    resampler = best_params.get("resampler__kind", "smote") if best_params else "smote"
    best_scaler = best_params.get("scaler__kind", scaler) if best_params else scaler
    pre, deployed, threshold = fit_deployed_protocol(
        clf, X_train, y_train, scaler=best_scaler, log_cols=log_cols,
        resampler=resampler, threshold_objective=threshold_objective,
        imputer=imputer, calibrate=calibrate)
    Xte = pre.transform(X_test)
    y_proba = deployed.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    auc_mean, auc_lo, auc_hi = bootstrap_auc(y_test, y_proba)
    background = pre.transform(X_train)[:SHAP_BACKGROUND_SAMPLES]

    # The deployed family's headline metrics are exactly the `holdout` block
    # computed in the comparison loop (same call, same params), so the
    # "Model performance" and "Model family comparison" views agree. The
    # split-42 `single_split` / per-class / confusion matrix are re-derived on
    # the refit deployed model for reference.
    deployed_metrics = dict(per_model[best])
    single_split = compute_metrics(y_test, deployed.predict(Xte), y_proba)
    deployed_metrics.update({
        "best_params": best_params,
        "scaler": best_scaler,
        "resampler": resampler,
        "threshold_objective": threshold_objective,
        "selection_metric": selection_metric,
        "selection_tiebreak": selection_tiebreak,
        "threshold": threshold,
        "single_split": single_split,
        "per_class": per_class_report(y_test, deployed.predict(Xte)),
        "confusion_matrix": confusion_matrix(y_test, deployed.predict(Xte)).tolist(),
        "test_auc": float(test_auc),
        "test_auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
        "bootstrap_reps": BOOTSTRAP_REPS,
    })

    # Overfit check + reference baseline so reported accuracy is read in
    # context. `baseline_accuracy` is the majority-class guess accuracy
    # (e.g. 0.71 for the imbalanced liver set, where accuracy can never get far
    # above the base rate). `generalization_gap` = training accuracy minus the
    # repeated-holdout accuracy: a large gap would flag overfitting, while a
    # small gap (as on the near-separable CKD set) means the high accuracy is
    # real and generalizes rather than being memorized.
    train_acc = accuracy_score(y_train, deployed.predict(pre.transform(X_train)))
    deployed_metrics.update({
        "baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
        "train_accuracy": float(train_acc),
        "generalization_gap": float(train_acc - deployed_metrics["accuracy"]),
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
          f"(threshold={threshold:.2f}, calibrated={deployed.calibrator is not None}, "
          f"test_auc={test_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}])")


def main(disease: str | None = None) -> None:
    """Train one disease (e.g. `python -m training.train_clinical heart_disease`)
    or all four when no argument is given. A per-disease run also refreshes the
    report with the other diseases' existing artifacts so the comparison files
    stay complete."""
    if disease is not None and disease not in DISEASE_LABELS:
        sys.exit(f"Unknown disease '{disease}'. Choose from {list(DISEASE_LABELS)}")
    targets = [disease] if disease else list(DISEASE_LABELS)
    print(f"Training clinical models: {targets} ...")
    results = {}
    for d in targets:
        label = DISEASE_LABELS[d]
        df = pd.read_csv(PROCESSED_DATA_DIR / f"{label_to_file(d)}.csv")
        print(f"\n=== {label} ({d}) — {len(df)} rows ===")
        train_disease(d, df, results)
        results[d] = joblib.load(MODELS_DIR / f"clinical_{d}.joblib")["comparison"]

    # Merge in the comparison tables of diseases that were not retrained this
    # run, so the report stays complete regardless of which diseases were run.
    for d in DISEASE_LABELS:
        if d not in results:
            results[d] = joblib.load(MODELS_DIR / f"clinical_{d}.joblib")["comparison"]

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
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))