"""TabICL AutoGluon AbstractModel wrapper.

Wraps ``tabicl.TabICLClassifier`` / ``TabICLRegressor`` for use in TabArena.
Model weights are resolved automatically from the local weights directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.common.utils.resource_utils import ResourceManager


class TabICLv2Model(AbstractModel):
    """TabICL in-context learning model for classification and regression."""

    ag_name = "TabICL-v2"
    ag_key = "TABICL_V2"
    _problem_types = {BINARY, MULTICLASS, REGRESSION}

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def _set_default_params(self) -> None:
        self._set_default_param_value("n_estimators", None)

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs) -> None:
        from torch.cuda import is_available
        device = "cuda" if num_gpus != 0 and is_available() else "cpu"
        n_estimators = self.params.get("n_estimators")

        if self.problem_type == REGRESSION:
            from tabicl import TabICLRegressor
            kw = {} if n_estimators is None else {"n_estimators": n_estimators}
            self.model = TabICLRegressor(device=device, n_jobs=num_cpus, **kw)
        else:
            from tabicl import TabICLClassifier
            kw = {} if n_estimators is None else {"n_estimators": n_estimators}
            self.model = TabICLClassifier(device=device, n_jobs=num_cpus, **kw)

        X = self.preprocess(X, y=y)
        self.model.fit(X=X, y=y)


class TabICL2reg(AbstractModel):
    """TabICL v2 Regressor used as a classifier.

    Fits ``TabICLRegressor`` on numeric-encoded class labels and converts
    regression outputs to class probabilities:

    - Binary:     regression on 0/1 labels → output clipped to [0, 1] as P(class=1).
    - Multiclass: regression on integer labels 0..K-1 → softmax over negative
                  distances to each class centroid.
    - Regression: plain regression (identical to TabICLv2Model).
    """

    ag_name = "TabICL2reg"
    ag_key = "TABICL2REG"
    _problem_types = {BINARY, MULTICLASS, REGRESSION}

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def _set_default_params(self) -> None:
        self._set_default_param_value("n_estimators", None)
        self._set_default_param_value("temperature", 1.0)

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs) -> None:
        from tabicl import TabICLRegressor
        from torch.cuda import is_available
        device = "cuda" if num_gpus != 0 and is_available() else "cpu"
        n_estimators = self.params.get("n_estimators")
        kw = {} if n_estimators is None else {"n_estimators": n_estimators}
        self.model = TabICLRegressor(device=device, n_jobs=num_cpus, **kw)
        if self.problem_type == MULTICLASS:
            self._n_classes = int(y.max()) + 1
        X = self.preprocess(X, y=y)
        self.model.fit(X=X, y=y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X)
        pred = self.model.predict(X)  # (n,) float

        if self.problem_type == REGRESSION:
            return pred
        elif self.problem_type == BINARY:
            return np.clip(pred, 0.0, 1.0)
        else:  # MULTICLASS
            from scipy.special import softmax
            temperature = self.params.get("temperature", 1.0)
            distances = np.abs(pred[:, None] - np.arange(self._n_classes))  # (n, K)
            return softmax(-distances / temperature, axis=1)

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        proba = self._predict_proba(X, **kwargs)
        if self.problem_type == REGRESSION:
            return proba
        elif self.problem_type == BINARY:
            return (proba >= 0.5).astype(int)
        else:  # MULTICLASS
            return proba.argmax(axis=1)
