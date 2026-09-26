"""TabPFN v2.6 thin wrapper (no autogluon)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TabPFNv26Model:
    name = "TabPFN-v2.6"

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = X.copy(); X.columns = X.columns.astype(str)

        if problem_type == "regression":
            from tabpfn_v2_6 import TabPFNRegressor
            self._model = TabPFNRegressor(ignore_pretraining_limits=True, device=device)
        else:
            from tabpfn_v2_6 import TabPFNClassifier
            self._model = TabPFNClassifier(ignore_pretraining_limits=True, device=device)
        self._model.fit(X=X, y=y)
        self._problem_type = problem_type

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict(X)
