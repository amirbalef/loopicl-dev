"""Thin wrappers for standard sklearn-compatible models.

All models share the same protocol:
    model.fit(X: pd.DataFrame, y: pd.Series, problem_type: str)
    model.predict_proba(X: pd.DataFrame) -> np.ndarray   # classification
    model.predict(X: pd.DataFrame) -> np.ndarray         # regression / any
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _preprocess(X: pd.DataFrame) -> np.ndarray:
    """Encode categoricals as ordinal codes (NaN → -1) and fill remaining NaN with 0."""
    out = X.copy()
    for col in out.columns:
        if out[col].dtype.name in ("object", "category"):
            codes = out[col].astype("category").cat.codes.astype(np.float32)
            out[col] = codes.where(codes != -1, other=np.nan)
    return out.fillna(0).to_numpy(dtype=np.float32)


class _SklearnModel:
    """Base for sklearn-compatible models."""
    name: str

    def __init__(self, clf_cls, reg_cls, clf_kwargs=None, reg_kwargs=None):
        self._clf_cls   = clf_cls
        self._reg_cls   = reg_cls
        self._clf_kwargs = clf_kwargs or {}
        self._reg_kwargs = reg_kwargs or {}

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        from sklearn.preprocessing import LabelEncoder
        X_arr = _preprocess(X)
        y_arr = y.to_numpy()
        if problem_type == "regression":
            self._model = self._reg_cls(**self._reg_kwargs)
            self._le = None
        else:
            # Label-encode y → 0-based integers (required by XGBoost; harmless for others)
            self._le = LabelEncoder()
            y_arr = self._le.fit_transform(y_arr)
            self._model = self._clf_cls(**self._clf_kwargs)
        self._model.fit(X_arr, y_arr)
        self._problem_type = problem_type

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(_preprocess(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(_preprocess(X))


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------

class LightGBMModel(_SklearnModel):
    name = "LightGBM"

    def __init__(self):
        from lightgbm import LGBMClassifier, LGBMRegressor
        super().__init__(
            LGBMClassifier, LGBMRegressor,
            clf_kwargs={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 128, "n_jobs": 1, "verbose": -1},
            reg_kwargs={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 128, "n_jobs": 1, "verbose": -1},
        )


class CatBoostModel(_SklearnModel):
    name = "CatBoost"

    def __init__(self):
        from catboost import CatBoostClassifier, CatBoostRegressor
        super().__init__(
            CatBoostClassifier, CatBoostRegressor,
            clf_kwargs={"iterations": 300, "learning_rate": 0.05, "verbose": 0},
            reg_kwargs={"iterations": 300, "learning_rate": 0.05, "verbose": 0},
        )


class XGBoostModel(_SklearnModel):
    name = "XGBoost"

    def __init__(self):
        from xgboost import XGBClassifier, XGBRegressor
        super().__init__(
            XGBClassifier, XGBRegressor,
            clf_kwargs={"n_estimators": 300, "learning_rate": 0.05, "n_jobs": 1,
                        "eval_metric": "logloss", "verbosity": 0},
            reg_kwargs={"n_estimators": 300, "learning_rate": 0.05, "n_jobs": 1, "verbosity": 0},
        )


class RandomForestModel(_SklearnModel):
    name = "RandomForest"

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        super().__init__(
            RandomForestClassifier, RandomForestRegressor,
            clf_kwargs={"n_estimators": 300, "n_jobs": 1},
            reg_kwargs={"n_estimators": 300, "n_jobs": 1},
        )


class ExtraTreesModel(_SklearnModel):
    name = "ExtraTrees"

    def __init__(self):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        super().__init__(
            ExtraTreesClassifier, ExtraTreesRegressor,
            clf_kwargs={"n_estimators": 300, "n_jobs": 1},
            reg_kwargs={"n_estimators": 300, "n_jobs": 1},
        )


# Registry: short name → factory
SKLEARN_MODELS: dict[str, type] = {
    "LightGBM":   LightGBMModel,
    "CatBoost":   CatBoostModel,
    "XGBoost":    XGBoostModel,
    "RandomForest": RandomForestModel,
    "ExtraTrees": ExtraTreesModel,
}
