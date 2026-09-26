"""Iris thin wrapper for OpenML benchmark (no autogluon).

Loads a trained Iris checkpoint and exposes it through the TabICL sklearn
interface, using the same fit/predict_proba/predict protocol as other openml models.

Usage in run_benchmark.py configs::

    # configs/iris_c1.py
    CONFIG = {"name": "Iris-c1", "model": "Iris", "weights_path": "/path/to/step.ckpt"}
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _load_iris_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    from ticla.models.iris.model import Iris

    checkpoint = torch.load(weights_path, map_location=device)
    model_config_dict = dict(checkpoint["config"])
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")
    model = Iris(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Calling-convention adapter
# ---------------------------------------------------------------------------


class _IrisForTabICL:
    """Adapts Iris to the calling convention expected by TabICLClassifier."""

    def __init__(self, iris: torch.nn.Module, max_train_samples: int | None = None):
        self._iris = iris
        self._max_train_samples = max_train_samples

    def __call__(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        return_logits: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        B, n_total = X.shape[0], X.shape[1]
        n_train = y_train.shape[1]

        if self._max_train_samples is not None and n_train > self._max_train_samples:
            idx = torch.randperm(n_train, device=X.device)[:self._max_train_samples].sort().values
            X = torch.cat([X[:, idx], X[:, n_train:]], dim=1)
            y_train = y_train[:, idx]
            n_train = self._max_train_samples
            n_total = X.shape[1]

        y_full = torch.zeros(B, n_total, dtype=y_train.dtype, device=y_train.device)
        y_full[:, :n_train] = y_train

        iris_kwargs: dict = {"single_eval_pos": n_train}
        if "num_loops" in kwargs:
            iris_kwargs["num_loops"] = kwargs["num_loops"]

        return self._forward_with_oom_retry(X, y_full, iris_kwargs)

    def _forward_with_oom_retry(self, X, y_full, iris_kwargs):
        device = X.device
        chunk_size = X.shape[0]

        while True:
            try:
                if X.device.type == device.type:
                    return self._iris(X, y_full, **iris_kwargs)
                else:
                    results = []
                    for start in range(0, X.shape[0], chunk_size):
                        end = min(start + chunk_size, X.shape[0])
                        out = self._iris(
                            X[start:end].to(device),
                            y_full[start:end].to(device),
                            **iris_kwargs,
                        )
                        results.append(out)
                    return torch.cat(results, dim=0)
            except torch.cuda.OutOfMemoryError as e:
                if chunk_size == 1:
                    raise
                e.__traceback__ = None
                del e
                X = X.cpu()
                y_full = y_full.cpu()
                for _ in range(3):
                    gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                print(f"[OOM] retrying with chunk_size={chunk_size}")

    def eval(self):
        self._iris.eval()
        return self

    def train(self, mode: bool = True):
        self._iris.train(mode)
        return self

    def to(self, *args, **kwargs):
        self._iris.to(*args, **kwargs)
        return self

    def parameters(self):
        return self._iris.parameters()


# ---------------------------------------------------------------------------
# Picklable _load_model replacement
# ---------------------------------------------------------------------------


class _IrisModelLoader:
    def __init__(self, clf, weights_path: str, num_loops, device: str, max_train_samples=None):
        self.clf = clf
        self.weights_path = weights_path
        self.num_loops = num_loops
        self.device = device
        self.max_train_samples = max_train_samples

    def __call__(self, *args, **kwargs):
        from torch.cuda import is_available
        from loopicl.interface.wrap_interface import _TabICLModelAdapter
        from loopicl.interface.wrap_model import TabICLWrapper

        device = "cuda" if is_available() else self.device
        iris = _load_iris_model(self.weights_path, device=device)
        adapter = _IrisForTabICL(iris, max_train_samples=self.max_train_samples)
        wrapper = TabICLWrapper(adapter, device, dtype="bfloat16", num_loops=self.num_loops)
        self.clf.model_ = _TabICLModelAdapter(wrapper)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class IrisModel:
    """Iris in-context learning model for binary / multiclass classification.

    Hyperparameters (pass via config)
    ----------------------------------
    weights_path : str
        Path to a trained Iris ``.ckpt`` checkpoint.
    interface : str, default ``"tabicl"``
        Which sklearn interface to wrap around the Iris model.
    n_estimators : int, optional
    num_loops : int, optional
    batch_size : int, default 8
    max_train_samples : int, optional
    subsample_samples : int, optional
    """

    name = "Iris"

    def __init__(
        self,
        weights_path: str | None = None,
        interface: str = "tabicl",
        n_estimators: int | None = None,
        num_loops: int | None = None,
        batch_size: int = 8,
        max_train_samples: int | None = None,
        subsample_samples: int | None = None,
    ):
        self._weights_path = weights_path
        self._interface = interface
        self._n_estimators = n_estimators
        self._num_loops = num_loops
        self._batch_size = batch_size
        self._max_train_samples = max_train_samples
        self._subsample_samples = subsample_samples

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch
        from loopicl.interface.wrap_interface import wrap_interface
        from loopicl.interface.wrap_model import TabICLWrapper

        if not self._weights_path:
            raise ValueError(
                "IrisModel requires a 'weights_path' pointing to a trained Iris checkpoint."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._problem_type = problem_type

        iris = _load_iris_model(self._weights_path, device=device)
        adapter = _IrisForTabICL(iris, max_train_samples=self._max_train_samples)
        wrapper = TabICLWrapper(adapter, device, dtype="bfloat16", num_loops=self._num_loops)

        if self._interface == "tabicl":
            clf = wrap_interface("tabicl", task_type="cls", model=wrapper, device=device,
                                 batch_size=self._batch_size)
            if self._n_estimators is not None:
                clf.n_estimators = self._n_estimators
            clf._load_model = _IrisModelLoader(
                clf, self._weights_path, self._num_loops, device, self._max_train_samples
            )
        elif self._interface == "ricotta":
            from loopicl.interface import LoopICLClassifier
            clf = LoopICLClassifier(
                wrapper=wrapper,
                n_estimators=self._n_estimators if self._n_estimators is not None else 8,
                batch_size=self._batch_size,
                subsample_samples=self._subsample_samples,
            )
        else:
            clf = wrap_interface(self._interface, task_type="cls", model=wrapper)
            if self._n_estimators is not None:
                clf.n_estimators = self._n_estimators
            clf.device = device

        X = X.copy(); X.columns = X.columns.astype(str)
        clf.fit(X=X, y=y)
        self._model = clf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)
