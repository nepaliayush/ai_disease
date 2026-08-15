"""Pydantic request/response schemas for the prediction API."""
from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    clinical: dict[str, Any] = Field(
        ..., description="Unified clinical parameter values (all fields required).")
    symptoms: list[str] = Field(
        default_factory=list,
        description="Selected symptom checklist entries (canonical symptom names).")


class ShapEntry(BaseModel):
    feature: str
    label: str
    value: Union[float, int, None] = None
    shap: float
    contribution_pct: float


class ShapExplanation(BaseModel):
    model_name: str
    base_value: float
    entries: list[ShapEntry]


class Breakdown(BaseModel):
    clinical_weight: float
    symptom_weight: float
    clinical_contribution: float
    symptom_contribution: float
    clinical_share_pct: float
    symptom_share_pct: float


class SymptomCondition(BaseModel):
    disease: str
    probability: float


class LifestyleAdjustment(BaseModel):
    smoking: float
    alcohol: float
    total: float


class Prevalence(BaseModel):
    source: float
    target: float


class DiseaseResult(BaseModel):
    disease: str
    label: str
    clinical_risk_raw: float
    clinical_pct_raw: float
    clinical_risk: float
    clinical_pct: float
    prevalence: Prevalence
    symptom_relevance: float
    symptom_pct: float
    fused_risk: float
    fused_pct: float
    risk_level: str
    breakdown: Breakdown
    lifestyle_adjustment: LifestyleAdjustment
    shap: ShapExplanation
    top_conditions: list[SymptomCondition]


class OverallAssessment(BaseModel):
    fused_avg: float
    fused_avg_pct: float
    risk_level: str
    note: str


class PredictResponse(BaseModel):
    overall: OverallAssessment
    diseases: list[DiseaseResult]
    methodology: dict[str, Any]
    disclaimer: str


class PredictError(BaseModel):
    detail: str