"""TabPFN v3 thin wrapper (no autogluon)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TabPFNv3Model:
    name = "TabPFN-v3"

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._problem_type = problem_type

        if problem_type == "regression":
            self._model = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
        else:
            self._model = TabPFNClassifier.create_default_for_version(ModelVersion.V3)

        self._model.device = device
        X = X.copy(); X.columns = X.columns.astype(str)
        self._model.fit(X=X, y=y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict(X)
