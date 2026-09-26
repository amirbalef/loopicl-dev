"""Dummy baseline for TabArena evaluation.

Wraps ``sklearn.dummy.DummyClassifier`` / ``DummyRegressor`` as a single
AutoGluon ``AbstractModel`` subclass that handles all problem types.

Serves as a sanity-check baseline: any real model should outperform it.

Hyperparameters
---------------
strategy : str
    Passed directly to sklearn.  Defaults to ``"prior"`` for classification
    and ``"mean"`` for regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel


class DummyModel(AbstractModel):
    """Dummy baseline that works for classification and regression."""

    ag_name = "Dummy"
    ag_key = "DUMMY"
    _problem_types = {BINARY, MULTICLASS, REGRESSION}

    def _set_default_params(self) -> None:
        if self.problem_type == REGRESSION:
            self._set_default_param_value("strategy", "mean")
        else:
            self._set_default_param_value("strategy", "prior")

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        y_np = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        strategy = self.params.get("strategy")
        if self.problem_type == REGRESSION:
            from sklearn.dummy import DummyRegressor
            self.model = DummyRegressor(strategy=strategy)
        else:
            from sklearn.dummy import DummyClassifier
            self.model = DummyClassifier(strategy=strategy)
        self.model.fit(self._preprocess(X), y_np)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.problem_type == REGRESSION:
            return self.model.predict(self._preprocess(X))
        proba = self.model.predict_proba(self._preprocess(X))
        if self.problem_type == BINARY:
            return proba[:, 1]
        return proba

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(self._preprocess(X))

    def _get_model_base_cls(self):
        return DummyModel
