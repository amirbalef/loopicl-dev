"""EXAONE-Tabular runner for the TALENT benchmark.

Bridges the TALENT (N, C, y) data format to the EXAONE-Tabular sklearn-style
interface and returns a TALENT-compatible RunResult.

Install:
    pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"

Evaluation protocol (matches TALENT's ICL evaluation):
  1. Fit data converter on the train split only (ordinal encoding, NaN imputation).
  2. Binary classification: tune decision threshold on val using train-only context.
  3. Final predictions: test split, with train+val as the in-context training set.
  4. Metrics match TALENT's base.Method.metric() output exactly.
  5. Multi-seed: reload model + refit for each seed; aggregate mean/std.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse shared helpers from ricotta_runner to keep metrics consistent.
from benchmark_eval.talent.models.ricotta_runner import (
    _TalentDataConverter,
    _build_y_info,
    _compute_metrics,
    _tune_threshold,
)


def run_exaone(
    train_val_data: Tuple,
    test_data: Tuple,
    info: Dict[str, Any],
    seed_num: int = 1,
    save_path: str = "",
    device: Optional[str] = None,
) -> Any:
    """Fit EXAONE-Tabular on a TALENT dataset and return a RunResult.

    Parameters
    ----------
    train_val_data, test_data, info
        As returned by ``TALENT.model.lib.data.get_dataset``.
    seed_num : int
        Number of independent seeds to run and average over.
    save_path : str
        Kept for API compatibility (not used).
    device : str or None
        Device string passed to ``from_pretrained()`` (e.g. ``"cuda:0"``).
        If None, defaults to ``"cuda"`` when available, else ``"cpu"``.
    """
    import torch
    from TALENT.api import RunResult

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    task_type = info.get("task_type", "binclass")
    is_regression = task_type == "regression"

    # -----------------------------------------------------------------------
    # Unpack TALENT data
    # -----------------------------------------------------------------------
    N_tv, C_tv, y_tv = train_val_data
    N_te, C_te, y_te = test_data

    N_train = N_tv["train"] if N_tv is not None else None
    N_val   = N_tv["val"]   if N_tv is not None else None
    C_train = C_tv["train"] if C_tv is not None else None
    C_val   = C_tv["val"]   if C_tv is not None else None
    N_test  = N_te["test"]  if N_te is not None else None
    C_test  = C_te["test"]  if C_te is not None else None

    if is_regression:
        y_train = y_tv["train"].ravel().astype(np.float32)
        y_val   = y_tv["val"].ravel().astype(np.float32)
        y_test  = y_te["test"].ravel().astype(np.float32)
    else:
        from sklearn.preprocessing import LabelEncoder
        _le = LabelEncoder()
        _le.fit(np.concatenate([
            y_tv["train"].ravel(), y_tv["val"].ravel(), y_te["test"].ravel()
        ]))
        y_train = _le.transform(y_tv["train"].ravel()).astype(np.int64)
        y_val   = _le.transform(y_tv["val"].ravel()).astype(np.int64)
        y_test  = _le.transform(y_te["test"].ravel()).astype(np.int64)

    # -----------------------------------------------------------------------
    # Feature conversion: categorical → ordinal float, NaN imputation on num
    # (fit on train only to avoid leakage)
    # -----------------------------------------------------------------------
    converter  = _TalentDataConverter().fit(N_train, C_train)
    X_train    = converter.transform(N_train, C_train)
    X_val      = converter.transform(N_val,   C_val)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    X_test     = converter.transform(N_test,  C_test)

    y_info = _build_y_info(y_trainval, task_type)

    # -----------------------------------------------------------------------
    # Multi-seed loop
    # -----------------------------------------------------------------------
    per_seed: List[Dict[str, Any]] = []
    fit_times: List[float] = []
    predict_times: List[float] = []
    metric_names: tuple = ()

    for seed in range(seed_num):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if is_regression:
            from exaonetabular import EXAONETabularRegressor
            model = EXAONETabularRegressor.from_pretrained(device=device)
        else:
            from exaonetabular import EXAONETabularClassifier
            model = EXAONETabularClassifier.from_pretrained(device=device)

        # ------------------------------------------------------------------
        # Binary: tune threshold on val using train-only context
        # ------------------------------------------------------------------
        threshold: Optional[float] = None
        if task_type == "binclass":
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)
            threshold = _tune_threshold(proba_val, y_val.astype(np.int64))

        # ------------------------------------------------------------------
        # Fit on full train+val, predict on test
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        model.fit(X_trainval, y_trainval)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        if is_regression:
            preds_raw  = model.predict(X_test)
            proba_test = np.asarray(preds_raw).reshape(-1, 1)
        else:
            proba_test = model.predict_proba(X_test)
        predict_time = time.perf_counter() - t1

        metrics, metric_names = _compute_metrics(
            proba_test,
            y_test.astype(np.int64 if not is_regression else np.float64),
            task_type,
            y_info,
            threshold,
        )

        per_seed.append({
            "seed":          seed,
            "loss":          float("nan"),
            "metrics":       tuple(float(x) for x in metrics),
            "metric_names":  metric_names,
            "fit_time":      fit_time,
            "predict_time":  predict_time,
            "proba":         proba_test,
            "threshold":     threshold,
            "predictions":   proba_test,
            "predict_proba": proba_test if not is_regression else None,
            "predict_labels": (
                (proba_test[:, 1] >= threshold).astype(int)
                if task_type == "binclass" and threshold is not None
                else (proba_test.argmax(axis=1) if not is_regression else None)
            ),
        })
        fit_times.append(fit_time)
        predict_times.append(predict_time)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------------------------
    metrics_matrix = np.array(
        [list(s["metrics"]) for s in per_seed], dtype=np.float64
    )
    m_mean = tuple(metrics_matrix.mean(axis=0).tolist())
    m_std  = tuple(metrics_matrix.std(axis=0).tolist())

    last = per_seed[-1]

    return RunResult(
        predictions=last["proba"],
        predict_proba=last["predict_proba"],
        predict_labels=last["predict_labels"],
        threshold=last["threshold"],
        loss=float("nan"),
        metrics=last["metrics"],
        metric_names=metric_names,
        fit_time=float(np.mean(fit_times)),
        predict_time=float(np.mean(predict_times)),
        loss_mean=float("nan"),
        loss_std=0.0,
        metrics_mean=m_mean,
        metrics_std=m_std,
        per_seed=per_seed,
        method_type="exaone",
        config={},
        save_path=save_path,
    )
