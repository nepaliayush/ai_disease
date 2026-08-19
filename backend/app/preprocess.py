"""Clinical data preprocessing shared by training and serving.

Pipeline per disease:
  1. Median/KNN imputation of missing values
  2. IQR outlier detection + clipping (winsorization at Q1-1.5*IQR, Q3+1.5*IQR)
  3. Optional log1p transform or Yeo-Johnson power transform for skewed features
  4. Feature scaling (Min-Max or Standard, configurable)
  5. Optional feature selection (mutual information)
  6. SMOTE oversampling of the minority class (training only)

All transformers learn their state from the training split only, so no
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ImputerChoice(BaseEstimator, TransformerMixin):
    """Wraps SimpleImputer(median) or KNNImputer; `kind` is searchable."""

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


class PowerTransformChoice(BaseEstimator, TransformerMixin):
    """Applies Yeo-Johnson power transform to a fixed subset of columns.
    Yeo-Johnson handles zeros and negatives (unlike Box-Cox) and finds the
    optimal lambda to make the distribution more Gaussian, which helps
    linear models and distance-based classifiers.

    `kind`:
      - "yeojohnson": Yeo-Johnson (default, handles all values)
      - "log1p": log1p (original behavior, for backward compatibility)
      - "none": passthrough
    """

    def __init__(self, columns: list[str], kind: str = "yeojohnson") -> None:
        self.columns = columns
        self.kind = kind
        self.idx_ = None
        self.pt_ = None

    def fit(self, X, y=None):
        names = list(getattr(X, "columns", range(X.shape[1])))
        self.idx_ = [i for i, c in enumerate(names) if c in self.columns]
        if self.kind == "yeojohnson" and self.idx_:
            from sklearn.preprocessing import PowerTransformer
            self.pt_ = PowerTransformer(method="yeojohnson", standardize=False)
            X_arr = np.asarray(X, dtype=float)
            self.pt_.fit(X_arr[:, self.idx_])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.idx_ and self.kind == "yeojohnson" and self.pt_ is not None:
            X = X.copy()
            X[:, self.idx_] = self.pt_.transform(X[:, self.idx_])
        elif self.idx_ and self.kind == "log1p":
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


class FeatureSelectorChoice(BaseEstimator, TransformerMixin):
    """Optional mutual-information-based feature selection.

    `kind`:
      - "mutual_info": SelectKBest with mutual_info_classif
      - "none": passthrough (keep all features)

    When `k` is None, keeps all features (passthrough). When `k` is an int,
    selects the top-k features by mutual information with the target.
    """

    def __init__(self, kind: str = "none", k: int | None = None) -> None:
        self.kind = kind
        self.k = k
        self.selector_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        if self.kind == "mutual_info" and self.k is not None and self.k < X.shape[1]:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            self.selector_ = SelectKBest(mutual_info_classif, k=self.k)
            self.selector_.fit(X, y)
            self.selected_indices_ = np.where(self.selector_.get_support())[0]
        else:
            self.selected_indices_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)[self.selected_indices_]
        return self.selected_indices_


@dataclass
class ClinicalPreprocessor:
    """Container for a fitted set of preprocessors, reusable at serving time."""

    imputer: ImputerChoice
    clip: IQRClipTransformer
    log: Log1pTransformer | PowerTransformChoice | None
    scaler: ScalerChoice
    features: list[str]
    selector: FeatureSelectorChoice | None = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        X = self.imputer.transform(X)
        X = self.clip.transform(X)
        if self.log is not None:
            X = self.log.transform(X)
        X = self.scaler.transform(X)
        if self.selector is not None:
            X = self.selector.transform(X)
        return X


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
                     imputer: str = "median",
                     transform_kind: str = "log1p",
                     feature_selection: str = "none",
                     feature_selection_k: int | None = None,
                     y: np.ndarray | None = None,
                     ) -> ClinicalPreprocessor:
    """Fit imputer, IQR clipper, power transform, scaler, and optional feature
    selector on training data only.

    `scaler` is "minmax" or "standard". `imputer` is "median" (default) or
    "knn". `log_cols` lists features to transform. `transform_kind` controls
    the transform: "log1p" (original), "yeojohnson" (power transform), or
    "none". `feature_selection` is "none" or "mutual_info"."""
    imputer_obj = ImputerChoice(imputer).fit(X)
    X_imp = imputer_obj.transform(X)
    clip = IQRClipTransformer().fit(X_imp)
    X_clip = clip.transform(X_imp)
    log = None
    if log_cols:
        if transform_kind == "yeojohnson":
            log = PowerTransformChoice(log_cols, kind="yeojohnson").fit(X_clip)
        else:
            log = Log1pTransformer(log_cols).fit(X_clip)
        X_scaled_in = log.transform(X_clip)
    else:
        X_scaled_in = X_clip
    scaler_obj = ScalerChoice(scaler).fit(X_scaled_in)
    selector = None
    if feature_selection == "mutual_info" and feature_selection_k is not None:
        selector = FeatureSelectorChoice(kind="mutual_info", k=feature_selection_k)
        selector.fit(scaler_obj.transform(X_scaled_in), y)
    return ClinicalPreprocessor(
        imputer=imputer_obj, clip=clip, log=log, scaler=scaler_obj,
        features=list(X.columns), selector=selector,
    )


def build_clinical_pipeline(clf, scaler: str = "minmax",
                            log_cols: list[str] | None = None,
                            resampler: str = "smote",
                            imputer: str = "median",
                            transform_kind: str = "log1p",
                            feature_selection: str = "none",
                            feature_selection_k: int | None = None,
                            ) -> ImbPipeline:
    """Full evaluation pipeline (for cross-validation): impute -> clip ->
    power_transform? -> scale -> feature_select? -> sample -> classifier.
    Oversampling runs inside every CV fold to avoid leakage."""
    steps = [
        ("imputer", ImputerChoice(imputer)),
        ("clip", IQRClipTransformer()),
    ]
    if log_cols:
        if transform_kind == "yeojohnson":
            steps.append(("power_transform", PowerTransformChoice(log_cols, kind="yeojohnson")))
        else:
            steps.append(("log", Log1pTransformer(log_cols)))
    steps += [
        ("scaler", ScalerChoice(scaler)),
    ]
    if feature_selection == "mutual_info" and feature_selection_k is not None:
        steps.append(("selector", FeatureSelectorChoice(kind="mutual_info", k=feature_selection_k)))
    steps += [
        ("resampler", SamplerChoice(resampler)),
        ("clf", clf),
    ]
    return ImbPipeline(steps=steps)