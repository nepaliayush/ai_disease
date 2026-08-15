"""Late (decision-level) fusion of clinical risk and symptom relevance.

fused_risk = w_clinical * clinical_risk + w_symptom * symptom_relevance

The weights default to 0.7 / 0.3 (clinical data is objective and directly
trained on each disease's clinical dataset; symptoms are subjective and
self-reported). Tuning these weights empirically on a validation set is noted
as future work in the README methodology.
"""
from __future__ import annotations

from app.config import FUSION_W, LIFESTYLE_ADJUSTMENT, RISK_BANDS


def fused_risk(clinical_risk: float, symptom_relevance: float) -> float:
    return FUSION_W["clinical"] * clinical_risk + FUSION_W["symptom"] * symptom_relevance


def prevalence_recalibrate(score: float, p_source: float, p_target: float) -> float:
    """Label-shift correction of a predicted probability.

    Maps a model probability estimated on a dataset with source prevalence
    `p_source` onto a target population with prevalence `p_target`. Monotonic,
    so model ranking is preserved.
    """
    score = min(1.0, max(0.0, float(score)))
    if p_source <= 0 or p_source >= 1 or p_target <= 0 or p_target >= 1:
        return score
    if score in (0.0, 1.0):
        return score
    w_pos = p_target / p_source
    w_neg = (1.0 - p_target) / (1.0 - p_source)
    corrected = (score * w_pos) / (score * w_pos + (1.0 - score) * w_neg)
    return min(1.0, max(0.0, float(corrected)))


def lifestyle_adjustment(disease: str, smoking: str, alcohol: str) -> dict:
    """Additive lifestyle modifier (rule-based heuristic, disclosed separately)."""
    per_disease = LIFESTYLE_ADJUSTMENT[disease]
    smoking_adj = per_disease["smoking_status"][smoking]
    alcohol_adj = per_disease["alcohol_consumption"][alcohol]
    return {
        "smoking": round(smoking_adj, 4),
        "alcohol": round(alcohol_adj, 4),
        "total": round(smoking_adj + alcohol_adj, 4),
    }


def apply_lifestyle(fused: float, adjustment: dict) -> float:
    """Add the lifestyle adjustment to the fused risk, clamped to [0, 1]."""
    return float(min(1.0, max(0.0, fused + adjustment["total"])))


def breakdown(clinical_risk: float, symptom_relevance: float) -> dict:
    clinical_contribution = FUSION_W["clinical"] * clinical_risk
    symptom_contribution = FUSION_W["symptom"] * symptom_relevance
    fused = clinical_contribution + symptom_contribution
    clinical_share = (100.0 * clinical_contribution / fused) if fused > 0 else 50.0
    return {
        "clinical_weight": FUSION_W["clinical"],
        "symptom_weight": FUSION_W["symptom"],
        "clinical_contribution": round(clinical_contribution, 4),
        "symptom_contribution": round(symptom_contribution, 4),
        "clinical_share_pct": round(clinical_share, 1),
        "symptom_share_pct": round(100.0 - clinical_share, 1),
    }


def risk_level(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "Low"


def overall_assessment(fused_scores: list[float]) -> dict:
    avg = float(sum(fused_scores) / len(fused_scores)) if fused_scores else 0.0
    return {
        "fused_avg": round(avg, 4),
        "fused_avg_pct": round(avg * 100, 1),
        "risk_level": risk_level(avg),
        "note": (
            "Simple average of the four per-disease fused risk scores "
            "(not the raw clinical scores)."
        ),
    }