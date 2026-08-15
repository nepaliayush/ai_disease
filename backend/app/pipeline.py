"""End-to-end prediction pipeline: clinical models + symptom triage + fusion.

For each disease:
    clinical_risk      = deployed clinical model's P(disease | clinical data)
    symptom_relevance  = weighted sum of triage probabilities over mapped classes
    fused_risk         = 0.7 * clinical_risk + 0.3 * symptom_relevance
Overall assessment is the simple average of the four fused risks.
"""
from __future__ import annotations

import numpy as np

from app.config import (
    DISEASE_LABELS, DISEASES, PREVALENCE_SOURCE, PREVALENCE_TARGET, SHAP_TOP_K,
)
from app.fusion import (
    apply_lifestyle, breakdown, fused_risk, lifestyle_adjustment, overall_assessment,
    prevalence_recalibrate, risk_level,
)
from app.model_store import (
    build_clinical_inputs, get_clinical_artifact, get_symptom_artifact,
    predict_clinical,
)
from app.shap_engine import explain, format_explanation
from app.symptoms import predict_distribution, symptom_relevance, top_conditions


def _raw_feature_values(disease: str, clinical_inputs: dict) -> dict:
    """Raw (encoded) input values keyed by model feature name, for SHAP display."""
    frame = clinical_inputs[disease]
    return {col: frame.iloc[0][col] for col in frame.columns}


def run_pipeline(payload: dict) -> dict:
    clinical_inputs = build_clinical_inputs(payload)
    symptom_art = get_symptom_artifact()
    sym_model, sym_features, sym_classes = (
        symptom_art["model"], symptom_art["features"], symptom_art["classes"])

    dist = predict_distribution(sym_model, sym_features, sym_classes, payload["symptoms"])
    clinical = payload["clinical"]
    smoking = clinical.get("smoking_status", "never")
    alcohol = clinical.get("alcohol_consumption", "none")
    fused_scores = []
    disease_results = []

    for disease in DISEASES:
        artifact = get_clinical_artifact(disease)
        frame = clinical_inputs[disease]

        clinical_risk_raw = predict_clinical(disease, frame)
        clinical_risk = prevalence_recalibrate(
            clinical_risk_raw, PREVALENCE_SOURCE[disease], PREVALENCE_TARGET[disease])
        relevance = symptom_relevance(dist, disease)
        fused = fused_risk(clinical_risk, relevance)
        lifestyle = lifestyle_adjustment(disease, smoking, alcohol)
        fused = apply_lifestyle(fused, lifestyle)

        # SHAP explanation for the clinical model's contribution.
        X_row = artifact["preprocessor"].transform(frame)[0]
        base_value, shap_values = explain(
            artifact["shap_model"], artifact["is_tree_based"], X_row, artifact["background"])
        raw_values = _raw_feature_values(disease, clinical_inputs)
        shap_expl = format_explanation(
            artifact["features"], raw_values, shap_values, base_value, top_k=SHAP_TOP_K)
        shap_expl["model_name"] = artifact["model_name"]

        fused_scores.append(fused)
        disease_results.append({
            "disease": disease,
            "label": DISEASE_LABELS[disease],
            "clinical_risk_raw": round(clinical_risk_raw, 4),
            "clinical_pct_raw": round(clinical_risk_raw * 100, 1),
            "clinical_risk": round(clinical_risk, 4),
            "clinical_pct": round(clinical_risk * 100, 1),
            "prevalence": {
                "source": PREVALENCE_SOURCE[disease],
                "target": PREVALENCE_TARGET[disease],
            },
            "symptom_relevance": round(relevance, 4),
            "symptom_pct": round(relevance * 100, 1),
            "fused_risk": round(fused, 4),
            "fused_pct": round(fused * 100, 1),
            "risk_level": risk_level(fused),
            "breakdown": breakdown(clinical_risk, relevance),
            "lifestyle_adjustment": lifestyle,
            "shap": shap_expl,
            "top_conditions": top_conditions(dist, k=3),
        })

    return {
        "overall": overall_assessment(fused_scores),
        "diseases": disease_results,
        "symptom_distribution": dist,
    }