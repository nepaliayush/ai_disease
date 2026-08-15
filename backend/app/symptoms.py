"""Symptom triage inference and per-disease relevance mapping."""
from __future__ import annotations

import numpy as np

from app.fields import SYMPTOM_DISEASE_MAP_CANONICAL


def build_symptom_vector(selected: list[str], features: list[str]) -> np.ndarray:
    """One-hot vector over the model's feature vocabulary; unchecked -> 0."""
    selected = set(selected or [])
    return np.array([1.0 if f in selected else 0.0 for f in features], dtype=float).reshape(1, -1)


def predict_distribution(model, features: list[str], classes: list[str],
                         selected: list[str]) -> dict[str, float]:
    X = build_symptom_vector(selected, features)
    proba = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(classes, proba)}


def symptom_relevance(distribution: dict[str, float], disease: str) -> float:
    """Weighted sum of the triage class probabilities mapped to the target
    disease category (probability the symptom pattern aligns with the
    category)."""
    mapped = SYMPTOM_DISEASE_MAP_CANONICAL[disease]
    return float(sum(weight * distribution.get(cls, 0.0)
                     for cls, weight in mapped.items()))


def top_conditions(distribution: dict[str, float], k: int = 3) -> list[dict]:
    top = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"disease": cls, "probability": float(p)} for cls, p in top]