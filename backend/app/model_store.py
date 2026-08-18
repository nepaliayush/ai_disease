"""Loads the trained model artifacts and exposes inference helpers."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from app.config import DISEASES, MODELS_DIR
from app.fields import (
    bmi_category, DATASET_FEATURES, FIELD_TO_MODEL_FEATURE, encode_field,
)


@lru_cache(maxsize=1)
def _clinical_artifacts() -> dict[str, dict]:
    artifacts = {}
    for disease in DISEASES:
        artifacts[disease] = joblib.load(MODELS_DIR / f"clinical_{disease}.joblib")
    return artifacts


@lru_cache(maxsize=1)
def _symptom_artifact() -> dict:
    return joblib.load(MODELS_DIR / "symptom_triage.joblib")


def get_clinical_artifact(disease: str) -> dict:
    return _clinical_artifacts()[disease]


def get_symptom_artifact() -> dict:
    return _symptom_artifact()


def build_clinical_inputs(payload: dict) -> dict[str, pd.DataFrame]:
    """Map the unified form payload onto each disease's model feature frame."""
    clinical = payload["clinical"]
    inputs = {}
    for disease in DISEASES:
        features = DATASET_FEATURES[disease]
        row = {}
        for field in FIELD_TO_MODEL_FEATURE:
            if disease in FIELD_TO_MODEL_FEATURE[field]:
                model_feature = FIELD_TO_MODEL_FEATURE[field][disease]
                row[model_feature] = encode_field(field, clinical[field])
        if disease == "diabetes":
            # Engineered interactions matching the training-time features.
            glucose = float(clinical.get("glucose", 0.0))
            bmi = float(clinical.get("bmi", 0.0))
            age = float(clinical.get("age", 0.0))
            pregnancies = float(clinical.get("pregnancies", 0.0))
            insulin = float(clinical.get("insulin", 0.0))
            row["glucose_bmi"] = glucose / bmi if bmi > 0 else float("nan")
            row["bmi_age"] = bmi * age
            row["age_preg"] = age * pregnancies
            row["glucose_insulin_ratio"] = (
                glucose / insulin if insulin > 0 else float("nan"))
            row["bmi_category"] = bmi_category(bmi)
        if disease == "liver_disease":
            # Engineered ratios matching the training-time features. Divisions
            # by zero become NaN, which the preprocessor median-imputes.
            ast = float(clinical.get("ast", 0.0))
            alt = float(clinical.get("alt", 0.0))
            tbil = float(clinical.get("total_bilirubin", 0.0))
            dbil = float(clinical.get("direct_bilirubin", 0.0))
            tprot = float(clinical.get("total_proteins", 0.0))
            alb = float(clinical.get("serum_albumin", 0.0))
            row["ast_alt_ratio"] = ast / alt if alt > 0 else float("nan")
            row["direct_bilirubin_ratio"] = dbil / tbil if tbil > 0 else float("nan")
            row["bilirubin_total"] = tbil + dbil
            row["albumin_fraction"] = alb / tprot if tprot > 0 else float("nan")
            row["alt_ast_product"] = alt * ast
        inputs[disease] = pd.DataFrame([row], columns=features)
    return inputs


def predict_clinical(disease: str, frame: pd.DataFrame) -> float:
    artifact = get_clinical_artifact(disease)
    pre = artifact["preprocessor"]
    X = pre.transform(frame)
    proba = artifact["model"].predict_proba(X)[0, 1]
    return float(proba)


def all_model_metrics() -> dict:
    return {d: get_clinical_artifact(d)["metrics"] for d in DISEASES}


def all_model_comparisons() -> dict:
    """Per-disease comparison of all five model families (CV + held-out)."""
    return {d: get_clinical_artifact(d)["comparison"] for d in DISEASES}


def deployed_model_names() -> dict[str, str]:
    return {d: get_clinical_artifact(d)["model_name"] for d in DISEASES}