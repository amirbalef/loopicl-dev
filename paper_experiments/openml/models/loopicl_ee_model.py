"""Early-exit Iris thin wrapper for OpenML benchmark (no autogluon).

Runs Iris with ``return_all_logits=True`` and applies an early-exit method
to select the exit loop for each test dataset.

Usage in run_benchmark.py configs::

    # configs/iris_ee.py
    CONFIGS = [
        {"name": "Iris-EE-Fixed3", "model": "IrisEE",
         "weights_path": "/path/to/step.ckpt", "ee_method": "fixed", "ee_param": 3},
        {"name": "Iris-EE-MaxSoft0.8", "model": "IrisEE",
         "weights_path": "/path/to/step.ckpt", "ee_method": "max_softmax", "ee_param": 0.8},
    ]

    # Run:
    uv run benchmark_eval/openml/run_benchmark.py --config iris_ee
"""
from __future__ import annotations

import gc
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .iris_model import _load_iris_model, _IrisForTabICL


# ---------------------------------------------------------------------------
# Sklearn-compatible early-exit classifier
# ---------------------------------------------------------------------------


class _EarlyExitClassifier:
    """Stores training context and runs early-exit inference at predict time."""

    def __init__(
        self,
        iris_adapter: _IrisForTabICL,
        device: str,
        dtype: str,
        ee_method,
        max_loops: Optional[int],
        max_train_samples: Optional[int],
    ):
        self._iris = iris_adapter
        self._device = device
        self._dtype = dtype
        self._ee_method = ee_method
        self._max_loops = max_loops
        self._max_train_samples = max_train_samples

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_EarlyExitClassifier":
        self._X_train = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self._y_train = y_arr.astype(np.int64)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_test = np.asarray(X_test, dtype=np.float32)
        X_train = self._X_train
        y_train = self._y_train

        if self._max_train_samples is not None and len(y_train) > self._max_train_samples:
            idx = np.random.choice(len(y_train), self._max_train_samples, replace=False)
            idx.sort()
            X_train = X_train[idx]
            y_train = y_train[idx]

        X_full = np.concatenate([X_train, X_test], axis=0)
        n_train = len(y_train)

        device = self._device
        X_t = torch.as_tensor(X_full, device=device).unsqueeze(0)
        y_t = torch.as_tensor(y_train.astype(np.float32), device=device).unsqueeze(0)

        kwargs: dict = {}
        if self._max_loops is not None:
            kwargs["num_loops"] = self._max_loops

        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(self._dtype)

        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None and "cuda" in device
                else _nullcontext()
            )
            with ctx:
                all_logits = self._forward_all_logits(X_t, y_t, n_train, kwargs)

        ee = self._ee_method
        if ee.per_token:
            exit_indices = ee.select_exit_loop(all_logits)
            logits_MBT = ee.gather_exit_logits_per_token(all_logits, exit_indices)
        else:
            exit_indices = ee.select_exit_loop(all_logits)
            logits_MBT = ee.gather_exit_logits(all_logits, exit_indices)

        logits_MT = logits_MBT[:, 0, :]
        proba = torch.softmax(logits_MT.float(), dim=-1).cpu().numpy()

        n_classes = len(self.classes_)
        if proba.shape[1] > n_classes:
            proba = proba[:, :n_classes]
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return proba / row_sums

    def _forward_all_logits(self, X_t, y_t, n_train, extra_kwargs) -> List[torch.Tensor]:
        iris = self._iris._iris
        B = X_t.shape[0]
        n_total = X_t.shape[1]
        y_full = torch.zeros(B, n_total, dtype=y_t.dtype, device=y_t.device)
        y_full[:, :n_train] = y_t
        return iris(X_t, y_full, single_eval_pos=n_train, return_all_logits=True, **extra_kwargs)


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Early-exit method builder
# ---------------------------------------------------------------------------


def _build_ee_method(ee_method: str, ee_param: float):
    from ticla.eval.methods import (
        FixedLoops,
        MaxSoftmaxExit, EntropyExit, MarginExit, StabilityExit,
        PerTokenMaxSoftmaxExit, PerTokenEntropyExit, PerTokenMarginExit,
        PerTokenStabilityExit,
    )

    _dispatch = {
        "fixed":                   lambda p: FixedLoops(int(p)),
        "max_softmax":             lambda p: MaxSoftmaxExit(float(p)),
        "entropy":                 lambda p: EntropyExit(float(p)),
        "margin":                  lambda p: MarginExit(float(p)),
        "stability":               lambda p: StabilityExit(float(p)),
        "per_token_max_softmax":   lambda p: PerTokenMaxSoftmaxExit(float(p)),
        "per_token_entropy":       lambda p: PerTokenEntropyExit(float(p)),
        "per_token_margin":        lambda p: PerTokenMarginExit(float(p)),
        "per_token_stability":     lambda p: PerTokenStabilityExit(float(p)),
    }

    if ee_method not in _dispatch:
        raise ValueError(
            f"Unknown ee_method {ee_method!r}. Choose from: {list(_dispatch)}"
        )
    return _dispatch[ee_method](ee_param)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class EarlyExitIrisModel:
    """Iris early-exit model for binary / multiclass classification.

    Hyperparameters (pass via config)
    ----------------------------------
    weights_path : str
        Path to a trained Iris ``.ckpt`` checkpoint.
    ee_method : str
        Early-exit strategy: ``"fixed"``, ``"max_softmax"``, ``"entropy"``,
        ``"margin"``, ``"stability"``, or per-token variants.
    ee_param : float
        Main hyperparameter for the chosen method.
    max_loops : int, optional
    batch_size : int, default 8  (kept for API parity, unused here)
    max_train_samples : int, optional
    """

    name = "IrisEE"

    def __init__(
        self,
        weights_path: str | None = None,
        ee_method: str = "fixed",
        ee_param: float = 6,
        max_loops: int | None = None,
        batch_size: int = 8,
        max_train_samples: int | None = None,
    ):
        self._weights_path = weights_path
        self._ee_method = ee_method
        self._ee_param = ee_param
        self._max_loops = max_loops
        self._max_train_samples = max_train_samples

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch

        if not self._weights_path:
            raise ValueError(
                "EarlyExitIrisModel requires a 'weights_path' pointing to a trained Iris checkpoint."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._problem_type = problem_type

        iris = _load_iris_model(self._weights_path, device=device)
        adapter = _IrisForTabICL(iris, max_train_samples=None)
        method_obj = _build_ee_method(self._ee_method, self._ee_param)

        clf = _EarlyExitClassifier(
            iris_adapter=adapter,
            device=device,
            dtype="bfloat16",
            ee_method=method_obj,
            max_loops=self._max_loops,
            max_train_samples=self._max_train_samples,
        )

        X = X.copy(); X.columns = X.columns.astype(str)
        clf.fit(X=np.asarray(X, dtype=np.float32), y=np.asarray(y))
        self._model = clf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)
