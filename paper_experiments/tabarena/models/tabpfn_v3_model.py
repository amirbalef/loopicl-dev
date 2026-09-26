"""TabPFN v3 AutoGluon AbstractModel wrapper.

Wraps ``tabpfn.TabPFNClassifier`` / ``TabPFNRegressor`` (ModelVersion.V3) for use in TabArena.
"""

from __future__ import annotations

import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.common.utils.resource_utils import ResourceManager


class TabPFNv3Model(AbstractModel):
    """TabPFN v3 in-context learning model for classification and regression."""

    ag_name = "TabPFN-v3"
    ag_key = "TABPFN_V3"
    _problem_types = {BINARY, MULTICLASS, REGRESSION}

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def _set_default_params(self) -> None:
        self._set_default_param_value("n_estimators", None)
        self._set_default_param_value("auto_scale_n_estimators", True)

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs) -> None:
        from torch.cuda import is_available
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        device = "cuda" if num_gpus != 0 and is_available() else "cpu"
        n_estimators = self.params.get("n_estimators")

        if self.problem_type == REGRESSION:
            self.model = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
        else:
            self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V3)

        self.model.device = device
        if n_estimators is not None:
            self.model.n_estimators = n_estimators
        self.model.auto_scale_n_estimators = self.params.get("auto_scale_n_estimators", True)

        X = self.preprocess(X, y=y)
        self.model.fit(X=X, y=y)
