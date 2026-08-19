"""Shared configuration: paths, fusion weights, risk thresholds."""
from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
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
FUSION_W = {"clinical": 0.7, "symptom": 0.3}

DISEASES = ["diabetes", "heart_disease", "liver_disease", "ckd"]

DISEASE_LABELS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "liver_disease": "Liver Disease",
    "ckd": "Chronic Kidney Disease",
}

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
    "CatBoost": (
        lambda: CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4,
            bootstrap_type="Bernoulli", verbose=0, allow_writing_files=False,
            random_seed=42,
        ),
        True,
    ),
    "MLP": (
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=42),
        False,
    ),
    "LightGBM": (
        lambda: _make_lgbm(),
        True,
    ),
}


def _make_lgbm():
    """Lazy import for LightGBM to avoid import errors if not installed."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        verbose=-1, importance_type="gain",
    )


# Per-disease model families.
BASE_CLINICAL_FAMILIES = ["LogisticRegression", "SVM", "RandomForest", "XGBoost",
                          "LightGBM", "MLP"]
EXTRA_CLINICAL_FAMILIES = {
    "heart_disease": ["CatBoost"],
}


def clinical_families(disease: str) -> list[str]:
    """Model family names compared/trained for a given disease."""
    return BASE_CLINICAL_FAMILIES + EXTRA_CLINICAL_FAMILIES.get(disease, [])

SHAP_BACKGROUND_SAMPLES = 100
SHAP_TOP_K = 10
SHAP_KERNEL_SAMPLES = 40

MODEL_SELECTION_METRIC = "cv_auc_mean"
MODEL_SELECTION_TIEBREAK = "roc_auc"

# ---------------------------------------------------------------------------
# Per-disease preprocessing choices.
# ---------------------------------------------------------------------------
LIVER_LOG_COLS = [
    "total_bilirubin", "direct_bilirubin", "alkaline_phosphotase",
    "alamine_aminotransferase", "aspartate_aminotransferase",
    "ast_alt_ratio", "direct_bilirubin_ratio", "bilirubin_total",
    "alt_ast_product", "bilirubin_alp", "injury_cholestasis_ratio",
]

DISEASE_PREPROCESSING = {
    "diabetes": {"scaler": "minmax", "log_cols": None, "threshold_objective": "f1",
                 "imputer": "knn"},
    "heart_disease": {"scaler": "minmax", "log_cols": None,
                      "threshold_objective": "f1",
                      "calibrate": True},
    "liver_disease": {"scaler": "minmax", "log_cols": LIVER_LOG_COLS,
                      "threshold_objective": "f1",
                      "imputer": "median",
                      "transform_kind": "yeojohnson",
                      "selection_metric": "cv_pr_auc_mean",
                      "selection_tiebreak": "pr_auc"},
    "ckd": {"scaler": "minmax", "log_cols": None, "threshold_objective": "f1"},
}

# ---------------------------------------------------------------------------
# Lifestyle risk adjustment.
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
# Prevalence-aware recalibration.
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