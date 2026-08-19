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
            glucose = float(row.get("glucose", 0.0))
            bmi = float(row.get("bmi", 0.0))
            age = float(row.get("age", 0.0))
            pregnancies = float(row.get("pregnancies", 0.0))
            insulin = float(row.get("insulin", 0.0))
            skin_thickness = float(row.get("skin_thickness", 0.0))
            pedigree = float(row.get("diabetes_pedigree_function", 0.0))
            row["glucose_bmi"] = glucose / bmi if bmi > 0 else float("nan")
            row["bmi_age"] = bmi * age
            row["age_preg"] = age * pregnancies
            row["glucose_insulin_ratio"] = (
                glucose / insulin if insulin > 0 else float("nan"))
            row["bmi_category"] = bmi_category(bmi)
            # Additional engineered features.
            row["homa_proxy"] = glucose * insulin
            row["bmi_age_risk"] = bmi * age * age
            row["glucose_age"] = glucose / age if age > 0 else float("nan")
            row["pedigree_age"] = pedigree * age
            row["preg_load"] = pregnancies * age
            row["skin_bmi"] = skin_thickness * bmi
        if disease == "heart_disease":
            age_val = float(row.get("age", 0))
            cp_val = float(row.get("cp", 0))
            chol_val = float(row.get("chol", 0))
            thalach_val = float(row.get("thalach", 0))
            exang_val = float(row.get("exang", 0))
            oldpeak_val = float(row.get("oldpeak", 0))
            trestbps_val = float(row.get("trestbps", 0))
            ca_val = float(row.get("ca", 0))
            thal_val = float(row.get("thal", 0))
            sex_val = float(row.get("sex", 0))
            row["age_cp"] = age_val * cp_val
            row["chol_thalach_ratio"] = (
                chol_val / thalach_val if thalach_val > 0 else float("nan"))
            row["oldpeak_exang"] = oldpeak_val * exang_val
            row["age_oldpeak"] = age_val * oldpeak_val
            row["hr_deficit"] = (220 - age_val) - thalach_val
            row["bp_chol"] = trestbps_val * chol_val
            row["ca_thal"] = ca_val * thal_val
            row["age_sex"] = age_val * sex_val
            row["chol_age"] = chol_val / age_val if age_val > 0 else float("nan")
        if disease == "liver_disease":
            ast = float(row.get("aspartate_aminotransferase", 0.0))
            alt = float(row.get("alamine_aminotransferase", 0.0))
            tbil = float(row.get("total_bilirubin", 0.0))
            dbil = float(row.get("direct_bilirubin", 0.0))
            tprot = float(row.get("total_proteins", 0.0))
            alb = float(row.get("albumin", 0.0))
            alp = float(row.get("alkaline_phosphotase", 0))
            age_l = float(row.get("age", 1))
            row["ast_alt_ratio"] = ast / alt if alt > 0 else float("nan")
            row["direct_bilirubin_ratio"] = dbil / tbil if tbil > 0 else float("nan")
            row["bilirubin_total"] = tbil + dbil
            row["albumin_fraction"] = alb / tprot if tprot > 0 else float("nan")
            row["alt_ast_product"] = alt * ast
            # Additional engineered features.
            row["globulin"] = tprot - alb
            row["alp_age"] = alp / age_l if age_l > 0 else float("nan")
            row["bilirubin_alp"] = tbil * alp
            row["injury_cholestasis_ratio"] = (
                (alt * ast) / alp if alp > 0 else float("nan"))
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