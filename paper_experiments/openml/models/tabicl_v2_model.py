"""TabICL v2 thin wrappers (no autogluon)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TabICLv2Model:
    name = "TabICL-v2"

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch
        from tabicl import TabICLClassifier, TabICLRegressor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = X.copy(); X.columns = X.columns.astype(str)

        if problem_type == "regression":
            self._model = TabICLRegressor(device=device)
        else:
            self._model = TabICLClassifier(device=device)
        self._model.fit(X=X, y=y)
        self._problem_type = problem_type

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict(X)
