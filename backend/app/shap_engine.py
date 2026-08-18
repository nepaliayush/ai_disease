"""SHAP explainability for the deployed clinical models.

Strategy per model family (kept fast for online serving):
  - tree models (Random Forest / XGBoost / CatBoost) -> TreeExplainer (interventional)
  - linear models (LogisticRegression, linear  -> LinearExplainer (exact)
    SVC / SVM)
  - anything else (MLP, RBF SVM)               -> KernelExplainer fallback on a
                                                 small background subsample
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from app.config import SHAP_KERNEL_SAMPLES


def _is_linear(model) -> bool:
    return isinstance(model, (LogisticRegression, LinearSVC))


def _pick(values, class_index: int):
    if isinstance(values, list):
        return values[class_index]
    return values


def explain(model, is_tree_based: bool, X_row: np.ndarray, background: np.ndarray,
            class_index: int = 1):
    """Return (expected_value, shap_values_for_class) for a single row."""
    X_row = np.asarray(X_row).reshape(1, -1)
    background = np.asarray(background)

    if is_tree_based:
        explainer = shap.TreeExplainer(model, data=background)
        ev = explainer.expected_value
        if isinstance(ev, np.ndarray) and ev.ndim == 1 and ev.shape[0] > 1:
            ev = float(ev[class_index])
        else:
            ev = float(np.asarray(ev).ravel()[0])
        vals = _pick(explainer.shap_values(X_row), class_index)
    elif _is_linear(model):
        explainer = shap.LinearExplainer(
            model, masker=shap.maskers.Independent(background))
        ev = float(np.asarray(explainer.expected_value).ravel()[0])
        vals = _pick(explainer.shap_values(X_row), class_index)
    else:
        bg = background[:SHAP_KERNEL_SAMPLES]
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        ev = float(np.asarray(explainer.expected_value).ravel()[0])
        vals = _pick(explainer.shap_values(X_row), class_index)

    return ev, np.asarray(vals).reshape(-1)


def format_explanation(features: list[str], raw_values: dict, shap_values: np.ndarray,
                       base_value: float, top_k: int = 10) -> dict:
    """Package SHAP values into the API response (top-k by magnitude)."""
    total = float(np.abs(shap_values).sum()) or 1.0
    entries = []
    for name, sval in zip(features, shap_values):
        entries.append({
            "feature": name,
            "label": _pretty_feature(name),
            "value": raw_values.get(name),
            "shap": float(sval),
            "contribution_pct": round(100.0 * abs(float(sval)) / total, 2),
        })
    entries.sort(key=lambda e: abs(e["shap"]), reverse=True)
    return {
        "base_value": float(base_value),
        "entries": entries[:top_k],
    }


def _pretty_feature(name: str) -> str:
    return name.replace("_", " ").title()