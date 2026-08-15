"""Clinical data preprocessing shared by training and serving.

Pipeline per disease:
  1. Median imputation of missing values
  2. IQR outlier detection + clipping (winsorization at Q1-1.5*IQR, Q3+1.5*IQR)
  3. SMOTE oversampling of the minority class (training only)
  4. Min-Max normalization to [0, 1]

IQR clipping and scaling bounds are learned on the training split only, so no
test/query information leaks. The same fitted transformer set is reused at
serving time by the FastAPI application.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class IQRClipTransformer(BaseEstimator, TransformerMixin):
    """Detects outliers via the Tukey IQR rule and clips them to the whiskers."""

    def __init__(self) -> None:
        self.lower_ = None
        self.upper_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.feature_names_ = list(getattr(X, "columns", range(X.shape[1])))
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        # Constant columns keep their natural range (no clipping).
        with np.errstate(divide="ignore", invalid="ignore"):
            self.lower_ = np.where(iqr > 0, q1 - 1.5 * iqr, -np.inf)
            self.upper_ = np.where(iqr > 0, q3 + 1.5 * iqr, np.inf)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


@dataclass
class ClinicalPreprocessor:
    """Container for a fitted set of preprocessors, reusable at serving time."""

    imputer: SimpleImputer
    clip: IQRClipTransformer
    scaler: MinMaxScaler
    features: list[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        X = self.imputer.transform(X)
        X = self.clip.transform(X)
        return self.scaler.transform(X)


class CalibratedClassifier:
    """Wraps a fitted base classifier with an optional Platt-scaled (sigmoid)
    calibrator fit on a held-out calibration subset, plus a decision threshold
    optimized on that same subset. Preserves the base model (which is trained
    with SMOTE) while yielding calibrated probabilities.

    `calibrator=None` simply passes the base probabilities through unchanged
    (used for linear models that need no scaling)."""

    def __init__(self, base, calibrator=None, threshold: float = 0.5) -> None:
        self.base = base
        self.calibrator = calibrator
        self.threshold = float(threshold)

    def predict_proba(self, X):
        p = np.asarray(self.base.predict_proba(X)[:, 1], dtype=float)
        if self.calibrator is not None:
            q = self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            q = p
        return np.column_stack((1.0 - q, q))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def fit_preprocessor(X: pd.DataFrame) -> ClinicalPreprocessor:
    """Fit imputer, IQR clipper and scaler on training data only."""
    imputer = SimpleImputer(strategy="median").fit(X)
    X_imp = imputer.transform(X)
    clip = IQRClipTransformer().fit(X_imp)
    X_clip = clip.transform(X_imp)
    scaler = MinMaxScaler().fit(X_clip)
    return ClinicalPreprocessor(
        imputer=imputer, clip=clip, scaler=scaler,
        features=list(X.columns),
    )


def build_clinical_pipeline(clf) -> ImbPipeline:
    """Full evaluation pipeline (for cross-validation): impute -> clip -> scale
    -> SMOTE -> classifier. SMOTE runs inside every CV fold to avoid leakage."""
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", IQRClipTransformer()),
            ("scaler", MinMaxScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", clf),
        ]
    )