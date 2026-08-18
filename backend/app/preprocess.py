"""Clinical data preprocessing shared by training and serving.

Pipeline per disease:
  1. Median imputation of missing values
  2. IQR outlier detection + clipping (winsorization at Q1-1.5*IQR, Q3+1.5*IQR)
  3. Optional log1p transform of heavily right-skewed features (liver)
  4. Feature scaling (Min-Max or Standard, configurable)
  5. SMOTE oversampling of the minority class (training only)

IQR clipping, scaling and log-transform bounds are learned on the training split
only, so no test/query information leaks. The same fitted transformer set is
reused at serving time by the FastAPI application.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ImputerChoice(BaseEstimator, TransformerMixin):
    """Wraps SimpleImputer(median) or KNNImputer; `kind` is searchable.

    KNN imputation is used for diabetes, where the physiologically-impossible
    zeros are real missingness and the engineered ratio features correlate with
    their raw inputs, so nearest-neighbour imputation is more informative than a
    column median."""

    def __init__(self, kind: str = "median") -> None:
        self.kind = kind

    def fit(self, X, y=None):
        if self.kind == "knn":
            from sklearn.impute import KNNImputer
            self.imputer_ = KNNImputer(n_neighbors=5)
        else:
            from sklearn.impute import SimpleImputer
            self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)


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


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Applies log1p to a fixed subset of columns (heavily right-skewed liver
    markers like bilirubin and liver enzymes). Other columns pass through.
    The column indices are fixed at construction, so no data-dependent state
    is learned (safe inside CV folds and at serving time)."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.idx_ = None

    def fit(self, X, y=None):
        names = list(getattr(X, "columns", range(X.shape[1])))
        self.idx_ = [i for i, c in enumerate(names) if c in self.columns]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.idx_:
            X = X.copy()
            X[:, self.idx_] = np.log1p(X[:, self.idx_])
        return X


class ScalerChoice(BaseEstimator, TransformerMixin):
    """Wraps MinMaxScaler or StandardScaler; `kind` is searchable in a grid."""

    def __init__(self, kind: str = "minmax") -> None:
        self.kind = kind

    def fit(self, X, y=None):
        Scaler = StandardScaler if self.kind == "standard" else MinMaxScaler
        self.scaler_ = Scaler().fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)


class SamplerChoice(BaseEstimator):
    """Wraps SMOTE or SMOTEENN; `kind` is searchable in a grid."""

    def __init__(self, kind: str = "smote") -> None:
        self.kind = kind

    def fit_resample(self, X, y):
        if self.kind == "smoteenn":
            from imblearn.combine import SMOTEENN
            self.sampler_ = SMOTEENN(random_state=42)
        else:
            self.sampler_ = SMOTE(random_state=42)
        return self.sampler_.fit_resample(X, y)

    def fit(self, X, y=None):
        return self


@dataclass
class ClinicalPreprocessor:
    """Container for a fitted set of preprocessors, reusable at serving time."""

    imputer: ImputerChoice
    clip: IQRClipTransformer
    log: Log1pTransformer | None
    scaler: ScalerChoice
    features: list[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        X = self.imputer.transform(X)
        X = self.clip.transform(X)
        if self.log is not None:
            X = self.log.transform(X)
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


def fit_preprocessor(X: pd.DataFrame, scaler: str = "minmax",
                     log_cols: list[str] | None = None,
                     imputer: str = "median") -> ClinicalPreprocessor:
    """Fit imputer, IQR clipper and scaler on training data only.

    `scaler` is "minmax" or "standard". `imputer` is "median" (default) or
    "knn". `log_cols` optionally log1p-transforms a fixed subset of columns
    (e.g. heavily skewed liver markers)."""
    imputer_obj = ImputerChoice(imputer).fit(X)
    X_imp = imputer_obj.transform(X)
    clip = IQRClipTransformer().fit(X_imp)
    X_clip = clip.transform(X_imp)
    log = None
    if log_cols:
        log = Log1pTransformer(log_cols).fit(X_clip)
        X_scaled_in = log.transform(X_clip)
    else:
        X_scaled_in = X_clip
    scaler_obj = ScalerChoice(scaler).fit(X_scaled_in)
    return ClinicalPreprocessor(
        imputer=imputer_obj, clip=clip, log=log, scaler=scaler_obj,
        features=list(X.columns),
    )


def build_clinical_pipeline(clf, scaler: str = "minmax",
                            log_cols: list[str] | None = None,
                            resampler: str = "smote",
                            imputer: str = "median") -> ImbPipeline:
    """Full evaluation pipeline (for cross-validation): impute -> clip -> log1p?
    -> scale -> sample (SMOTE/SMOTEENN) -> classifier. Oversampling runs inside
    every CV fold to avoid leakage."""
    steps = [
        ("imputer", ImputerChoice(imputer)),
        ("clip", IQRClipTransformer()),
    ]
    if log_cols:
        steps.append(("log", Log1pTransformer(log_cols)))
    steps += [
        ("scaler", ScalerChoice(scaler)),
        ("resampler", SamplerChoice(resampler)),
        ("clf", clf),
    ]
    return ImbPipeline(steps=steps)