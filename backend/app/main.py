"""FastAPI application: multi-model disease prediction with late fusion.

Endpoints:
  GET  /            -> service info
  GET  /metadata    -> form schema (sections, fields, symptoms, diseases, weights)
  GET  /health      -> readiness
  POST /predict     -> fused risk assessment for all four diseases

CORS is enabled so the Next.js frontend (localhost:3000) can call the API.
Run:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    DISEASES, DISEASE_LABELS, FUSION_W, LIFESTYLE_NOTE, PREVALENCE_NOTE,
)
from app.fields import (
    ENCODERS, FIELD_TO_MODEL_FEATURE, SYMPTOM_CHECKLIST, SYMPTOM_DISCLAIMER,
    UNIFIED_FIELDS,
)
from app.model_store import (
    all_model_comparisons, all_model_metrics, deployed_model_names,
)
from app.pipeline import run_pipeline
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Multi-Model Disease Prediction API",
    version="1.0.0",
    description=(
        "Fuses four clinical ML models (diabetes, heart, liver, CKD) with a "
        "symptom triage model using decision-level fusion."
    ),
)

# CORS origins: allow the local Next.js dev server on any port (common in
# development), plus explicit origins from the CORS_ORIGINS env var
# (comma-separated) for other deployments.
_extra = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_extra,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\]):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "disease-prediction",
        "endpoints": ["/metadata", "/health", "/predict"],
        "models": deployed_model_names(),
        "fusion_weights": FUSION_W,
    }


@app.get("/health")
def health() -> dict:
    try:
        all_model_metrics()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Models not ready: {exc}")


@app.get("/metadata")
def metadata() -> dict:
    return {
        "diseases": [
            {"id": d, "label": DISEASE_LABELS[d]} for d in DISEASES
        ],
        "sections": [
            {"title": title, "fields": fields} for title, fields in UNIFIED_FIELDS.items()
        ],
        "symptoms": SYMPTOM_CHECKLIST,
        "fusion_weights": FUSION_W,
        "deployed_models": deployed_model_names(),
        "model_metrics": all_model_metrics(),
        "model_comparison": all_model_comparisons(),
        "disclaimer": SYMPTOM_DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    _validate_payload(request)
    result = run_pipeline(request.model_dump())
    result["methodology"] = {
        "fusion_formula": "fused_risk = 0.7 * clinical_risk + 0.3 * symptom_relevance",
        "fusion_weights": FUSION_W,
        "weight_justification": (
            "Clinical data is objective and machine-measured and each clinical "
            "model is trained on its own disease dataset, so it carries the "
            "larger weight. Symptoms are subjective and self-reported, hence the "
            "smaller weight. The 0.7/0.3 split is a reasoned default; empirical "
            "tuning on a held-out validation set is noted as future work."
        ),
        "overall_assessment": (
            "Simple average of the four fused risk scores (not raw clinical scores)."
        ),
        "clinical_models": deployed_model_names(),
        "shap_method": (
            "Interventional TreeExplainer (tree models) / LinearExplainer "
            "(linear models) over a background sample from the training split."
        ),
        "lifestyle_adjustment": LIFESTYLE_NOTE,
        "prevalence_recalibration": PREVALENCE_NOTE,
    }
    result["disclaimer"] = (
        "This tool is for research and education only. It is not a medical device "
        "and does not provide a diagnosis. Consult a qualified clinician for any "
        "health decisions. " + SYMPTOM_DISCLAIMER
    )
    return result


def _validate_payload(request: PredictRequest) -> None:
    required = {f["name"] for section in UNIFIED_FIELDS.values() for f in section}
    missing = sorted(required - set(request.clinical.keys()))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing clinical fields: {missing}")

    for name, value in request.clinical.items():
        if name not in required:
            raise HTTPException(status_code=422, detail=f"Unknown clinical field: {name}")
        if name in ENCODERS:
            if value not in ENCODERS[name]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value '{value}' for field '{name}'",
                )

    unknown_symptoms = sorted(set(request.symptoms) - set(SYMPTOM_CHECKLIST))
    if unknown_symptoms:
        raise HTTPException(
            status_code=422, detail=f"Unknown symptoms: {unknown_symptoms}")