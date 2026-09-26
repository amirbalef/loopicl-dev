"""Dummy baseline (sklearn DummyClassifier / DummyRegressor)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DummyModel:
    name = "Dummy"

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        from models.sklearn_models import _preprocess
        X_arr = _preprocess(X)
        y_arr = y.to_numpy()
        if problem_type == "regression":
            from sklearn.dummy import DummyRegressor
            self._model = DummyRegressor(strategy="mean")
        else:
            from sklearn.dummy import DummyClassifier
            self._model = DummyClassifier(strategy="prior")
        self._model.fit(X_arr, y_arr)
        self._problem_type = problem_type

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        from models.sklearn_models import _preprocess
        return self._model.predict_proba(_preprocess(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        from models.sklearn_models import _preprocess
        return self._model.predict(_preprocess(X))
