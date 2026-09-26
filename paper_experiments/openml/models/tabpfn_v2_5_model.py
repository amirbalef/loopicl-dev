"""TabPFN v2.5 thin wrapper (no autogluon)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TabPFNv25Model:
    name = "TabPFN-v2.5"

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = X.copy(); X.columns = X.columns.astype(str)

        if problem_type == "regression":
            from tabpfn_v2_5 import TabPFNRegressor, TabPFN_v2_5_reg
            self._model = TabPFNRegressor(
                model_path=TabPFN_v2_5_reg, ignore_pretraining_limits=True, device=device
            )
        else:
            from tabpfn_v2_5 import TabPFNClassifier, TabPFN_v2_5
            self._model = TabPFNClassifier(
                model_path=TabPFN_v2_5, ignore_pretraining_limits=True, device=device
            )
        self._model.fit(X=X, y=y)
        self._problem_type = problem_type

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict(X)
