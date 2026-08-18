"""Shared configuration: paths, fusion weights, risk thresholds."""
from __future__ import annotations

from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Late (decision-level) fusion weights.
# Clinical measurements are objective / machine-measured and directly trained on
# each disease's own clinical dataset, so they receive the larger weight.
# Symptoms are subjective / self-reported, hence the smaller weight.
# This 0.7 / 0.3 split is a reasoned default; empirical tuning on a held-out
# validation set is flagged as future work (see README methodology section).
FUSION_W = {"clinical": 0.7, "symptom": 0.3}

DISEASES = ["diabetes", "heart_disease", "liver_disease", "ckd"]

DISEASE_LABELS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "liver_disease": "Liver Disease",
    "ckd": "Chronic Kidney Disease",
}

# Risk bands for the fused probability (applied per-disease and to the overall
# average).
RISK_BANDS = [
    (0.80, "Very High"),
    (0.60, "High"),
    (0.40, "Moderate"),
    (0.20, "Elevated"),
    (0.00, "Low"),
]

# Clinical models to compare (name -> (estimator factory, is_tree_based)).
CLINICAL_MODELS = {
    "LogisticRegression": (lambda: LogisticRegression(max_iter=3000), False),
    "SVM": (
        lambda: CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale"), ensemble=False),
        False,
    ),
    "RandomForest": (
        lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        True,
    ),
    "XGBoost": (
        lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            enable_categorical=False, random_state=42,
        ),
        True,
    ),
    "MLP": (
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=42),
        False,
    ),
}

# Number of background samples used to fit SHAP explainers at serving time.
SHAP_BACKGROUND_SAMPLES = 100
# Number of top features returned per disease in the SHAP explanation.
SHAP_TOP_K = 10
# Number of SHAP values returned if a slow (KernelExplainer) path is required.
SHAP_KERNEL_SAMPLES = 40

# Metric used to pick the deployed clinical model. CV-AUC (cross-validated ROC
# AUC via the leakage-free pipeline) is a more honest generalization signal
# than a single held-out test metric on these small datasets.
MODEL_SELECTION_METRIC = "cv_auc_mean"  # metric used to pick the deployed clinical model
MODEL_SELECTION_TIEBREAK = "roc_auc"  # tiebreak for model selection

# ---------------------------------------------------------------------------
# Per-disease preprocessing choices.
# `scaler` is "minmax" or "standard". `log_cols` lists features that receive a
# log1p transform before scaling (heavily right-skewed markers, e.g. liver
# enzymes/bilirubin). `threshold_objective` is what the deployed decision
# threshold is optimized for: "f1" (default) or "balanced_accuracy" (used for
# the imbalanced liver set so the deployed model does not just predict the
# majority class).
# ---------------------------------------------------------------------------
LIVER_LOG_COLS = [
    "total_bilirubin", "direct_bilirubin", "alkaline_phosphotase",
    "alamine_aminotransferase", "aspartate_aminotransferase",
    "ast_alt_ratio", "direct_bilirubin_ratio", "bilirubin_total",
    "alt_ast_product",
]

DISEASE_PREPROCESSING = {
    "diabetes": {"scaler": "minmax", "log_cols": None, "threshold_objective": "f1"},
    "heart_disease": {"scaler": "minmax", "log_cols": None, "threshold_objective": "f1"},
    # Liver is imbalanced (71% positive). A balanced-accuracy threshold was
    # tested and maximized per-class balance but dropped overall accuracy to
    # ~65% and recall to ~58% (missing most diseased patients). The F1
    # objective keeps the best accuracy (~72%) while SMOTE + class_weight
    # balanced already counter the majority-class bias.
    "liver_disease": {"scaler": "minmax", "log_cols": LIVER_LOG_COLS,
                      "threshold_objective": "f1"},
    "ckd": {"scaler": "minmax", "log_cols": None, "threshold_objective": "f1"},
}

# ---------------------------------------------------------------------------
# Lifestyle risk adjustment (transparent, rule-based, additive).
# The clinical training datasets do NOT contain smoking or alcohol features, so
# these fields cannot feed the models directly. Instead they are applied as an
# explicit, conservative additive modifier on top of the fused risk, clamped to
# [0, 1], and fully disclosed in the response and UI. These deltas are reasoned
# heuristics from published risk-factor evidence, NOT model outputs.
# ---------------------------------------------------------------------------
LIFESTYLE_ADJUSTMENT = {
    "diabetes": {
        "smoking_status": {"never": 0.00, "occasional": 0.01, "daily": 0.03},
        "alcohol_consumption": {"none": 0.00, "light": 0.00, "moderate": 0.02, "heavy": 0.03},
    },
    "heart_disease": {
        "smoking_status": {"never": 0.00, "occasional": 0.02, "daily": 0.04},
        "alcohol_consumption": {"none": 0.00, "light": 0.00, "moderate": 0.03, "heavy": 0.08},
    },
    "liver_disease": {
        "smoking_status": {"never": 0.00, "occasional": 0.01, "daily": 0.02},
        "alcohol_consumption": {"none": 0.00, "light": 0.02, "moderate": 0.06, "heavy": 0.12},
    },
    "ckd": {
        "smoking_status": {"never": 0.00, "occasional": 0.01, "daily": 0.02},
        "alcohol_consumption": {"none": 0.00, "light": 0.00, "moderate": 0.02, "heavy": 0.04},
    },
}

LIFESTYLE_NOTE = (
    "The clinical datasets used to train the four models do not include smoking "
    "or alcohol use, so these are applied as an explicit, conservative additive "
    "risk modifier on top of the fused score (rule-based heuristic, not model "
    "output) and shown separately as 'Lifestyle adjustment'."
)

# ---------------------------------------------------------------------------
# Prevalence-aware recalibration (label-shift correction).
# The clinical datasets are heavily enriched in positive cases (e.g. the ILPD
# liver dataset is 71% positive, CKD 63%), which inflates the models' predicted
# probabilities for everyone — including healthy people. We correct each
# clinical probability with the standard label-shift formula (Saerens et al.,
# 2002) so that an average member of the *target* population maps to the target
# prevalence while ranking is preserved:
#
#   w_pos = p_target / p_source,   w_neg = (1 - p_target) / (1 - p_source)
#   p_corrected = (p * w_pos) / (p * w_pos + (1 - p) * w_neg)
#
# `source` = positive prevalence measured in the processed training dataset.
# `target` = reasoned general-population prevalence (approximate, literature-
# based; age-adjusted diagnosis rates differ). Values are disclosed in the API
# and UI; they are tuning constants, not learned parameters.
# ---------------------------------------------------------------------------
PREVALENCE_SOURCE = {
    "diabetes": 0.3490, "heart_disease": 0.4587, "liver_disease": 0.7136, "ckd": 0.6250,
}

PREVALENCE_TARGET = {
    "diabetes": 0.10, "heart_disease": 0.10, "liver_disease": 0.10, "ckd": 0.12,
}

PREVALENCE_NOTE = (
    "The training datasets are enriched in positive cases (source prevalence: "
    "diabetes 35%, heart 46%, liver 71%, CKD 63%), which inflates raw model "
    "probabilities even for healthy people. Probabilities are therefore "
    "recalibrated to general-population base rates (diabetes/heart/liver ~10%, "
    "CKD ~12%) via label-shift correction (Saerens et al., 2002). Ranking is "
    "preserved; this is a calibration fix, not a diagnosis. SHAP values are "
    "computed on the raw clinical model and remain directionally valid because "
    "the correction is monotonic."
)