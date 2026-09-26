"""LoopICL runner for the TALENT benchmark.

Bridges the TALENT (N, C, y) data format to the LoopICLClassifier /
RicottaRegressor sklearn interface and returns a TALENT-compatible RunResult.

Evaluation protocol (matches TALENT's ICL evaluation):
  1. Fit data converter on the train split only (ordinal encoding, NaN imputation).
  2. Binary classification: tune decision threshold on val using train-only context.
  3. Final predictions: test split, with train+val as the in-context training set.
  4. Metrics match TALENT's base.Method.metric() output exactly.
  5. Multi-seed: re-load model + re-fit for each seed; aggregate mean/std.

Data format (TALENT output from get_dataset):
    train_val_data = (N_trainval, C_trainval, y_trainval)
        N_trainval : dict{'train': float array, 'val': float array} or None
        C_trainval : dict{'train': str array,   'val': str array}   or None
        y_trainval : dict{'train': int/float,   'val': int/float}
    test_data = (N_test, C_test, y_test)
        N_test     : dict{'test': float array} or None
        C_test     : dict{'test': str array}   or None
        y_test     : dict{'test': int/float}
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sklearn.metrics as skm
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _load_ricotta_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICL classification checkpoint.

    Loads to CPU first (avoids 2× VRAM spike), then moves to device.
    Prefers EMA weights when available.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])

    # Compat: older checkpoints use 'max_classes' instead of 'max_num_classes'
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")

    from loopicl.models.loopicl.model import LoopICL, LoopICLConfig
    known = set(LoopICLConfig.__dataclass_fields__)
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in known}

    model = LoopICL(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


def _load_ricotta_early_exit_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICLEarlyExit checkpoint (from TICLA).

    Loads to CPU first (avoids 2× VRAM spike), then moves to device.
    Prefers EMA weights when available.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])

    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")

    from loopicl.models.loopicl.model_early_exit import LoopICLEarlyExit, LoopICLEarlyExitConfig
    known = set(LoopICLEarlyExitConfig.__dataclass_fields__)
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in known}

    model = LoopICLEarlyExit(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


def _load_ricotta_early_exit_v2_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICLEarlyExitV2 checkpoint.

    Loads to CPU first (avoids 2× VRAM spike), then moves to device.
    Prefers EMA weights when available.
    """
    import os
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"LoopICLEarlyExitV2 checkpoint not found: {weights_path!r}\n"
            "Update 'weights_path' in your config file to point at a valid .ckpt file."
        )

    from loopicl.models.loopicl.model_early_exit_v2 import LoopICLEarlyExitV2, LoopICLEarlyExitV2Config

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")
    known = set(LoopICLEarlyExitV2Config.__dataclass_fields__)
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in known}
    model = LoopICLEarlyExitV2(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


def _load_ricotta_reg_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICLReg (quantile regression) checkpoint.

    Loads to CPU first (avoids 2× VRAM spike), then moves to device.
    Prefers EMA weights when available.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])

    # Compat: older checkpoints use 'max_classes' instead of 'max_num_classes'
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")

    from loopicl.models.loopicl.model import LoopICLConfig
    from loopicl.models.loopicl.model_reg import LoopICLReg

    _REG_KEYS = {"num_quantiles", "mlp_hidden_dim", "use_rank_label_repr"}
    base_known = set(LoopICLConfig.__dataclass_fields__)
    reg_kwargs  = {k: v for k, v in model_config_dict.items() if k in _REG_KEYS}
    base_kwargs = {k: v for k, v in model_config_dict.items() if k in base_known}

    model = LoopICLReg(**base_kwargs, **reg_kwargs)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Calling-convention adapter
# ---------------------------------------------------------------------------

class _RicottaAdapter:
    """Adapts LoopICL.forward to the TabICLWrapper calling convention.

    TabICLWrapper calls ``model(X, y_train, return_logits=True)`` where:
        X       : (B, n_total, n_features)
        y_train : (B, n_train)         ← only train labels

    LoopICL.forward expects:
        X       : (B, n_total, n_features)
        y_train : (B, n_total)         ← all rows (test rows are zeroed)
        single_eval_pos : int

    This adapter pads y_train with zeros and calls forward accordingly.
    OOM is handled by halving the batch chunk size until chunk=1.
    """

    def __init__(
        self,
        ricotta: torch.nn.Module,
        num_loops: Optional[int] = None,
        test_chunk_size: Optional[int] = None,
        exit_q: Optional[float] = None,
        heuristic_exit_strategy: Optional[str] = None,
        heuristic_exit_threshold: Optional[float] = None,
    ) -> None:
        self._ricotta = ricotta
        self._num_loops = num_loops
        self._test_chunk_size = test_chunk_size
        self._exit_q = exit_q
        self._heuristic_exit_strategy = heuristic_exit_strategy
        self._heuristic_exit_threshold = heuristic_exit_threshold
        self.max_classes = ricotta.cfg.max_num_classes
        # Persisted across calls: once a safe batch chunk size is found after
        # OOM, reuse it on subsequent calls instead of retrying from full size.
        self._safe_batch_chunk: Optional[int] = None

    def __call__(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        return_logits: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        B, n_total = X.shape[0], X.shape[1]
        n_train = y_train.shape[1]

        y_full = torch.zeros(B, n_total, dtype=y_train.dtype, device=y_train.device)
        y_full[:, :n_train] = y_train

        kw: dict = {"single_eval_pos": n_train}
        if self._num_loops is not None:
            kw["num_loops"] = self._num_loops
        elif "num_loops" in kwargs:
            kw["num_loops"] = kwargs["num_loops"]
        # Always enable test-row chunking: use configured value or default 512.
        kw["test_chunk_size"] = self._test_chunk_size if self._test_chunk_size is not None else 512
        if self._exit_q is not None:
            kw["exit_q"] = self._exit_q
        if self._heuristic_exit_strategy is not None:
            kw["heuristic_exit_strategy"] = self._heuristic_exit_strategy
        if self._heuristic_exit_threshold is not None:
            kw["heuristic_exit_threshold"] = self._heuristic_exit_threshold

        return self._forward_oom_safe(X, y_full, kw)

    def _calibrate_chunk_size(self, X: torch.Tensor, y_full: torch.Tensor, kw: dict) -> int:
        """Run one estimator, measure memory delta, derive safe batch chunk size."""
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info(X.device.index or 0)
        with torch.no_grad():
            _ = self._ricotta(X[:1], y_full[:1], **kw)
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info(X.device.index or 0)
        mem_per = free_before - free_after
        if mem_per <= 0:
            return X.shape[0]
        # Use 85% of remaining free memory to leave headroom.
        safe = max(1, int(free_after * 0.85 // mem_per))
        safe = min(safe, X.shape[0])
        print(
            f"    [mem] {free_before/1e9:.1f}→{free_after/1e9:.1f} GB free, "
            f"{mem_per/1e6:.0f} MB/est → batch_chunk={safe}"
        )
        return safe

    def _forward_oom_safe(
        self, X: torch.Tensor, y_full: torch.Tensor, kw: dict
    ) -> torch.Tensor:
        # Calibrate once on the first CUDA call to pick a safe chunk size.
        if self._safe_batch_chunk is None and X.device.type == "cuda":
            self._safe_batch_chunk = self._calibrate_chunk_size(X, y_full, kw)

        chunk_size = min(self._safe_batch_chunk or X.shape[0], X.shape[0])
        if chunk_size >= X.shape[0]:
            return self._ricotta(X, y_full, **kw)
        parts = []
        for i in range(0, X.shape[0], chunk_size):
            parts.append(self._ricotta(X[i:i + chunk_size], y_full[i:i + chunk_size], **kw))
        return torch.cat(parts, dim=0)

    def eval(self) -> "_RicottaAdapter":
        self._ricotta.eval()
        return self

    def train(self, mode: bool = True) -> "_RicottaAdapter":
        self._ricotta.train(mode)
        return self

    def to(self, *args, **kwargs) -> "_RicottaAdapter":
        self._ricotta.to(*args, **kwargs)
        return self

    def parameters(self):
        return self._ricotta.parameters()


# ---------------------------------------------------------------------------
# Data conversion: TALENT (N, C, y) → float32 numpy
# ---------------------------------------------------------------------------

class _TalentDataConverter:
    """Convert TALENT split data to a single float32 feature matrix.

    Fits on the train split only, then applies the same transformation
    to val, test, or train+val.

    Steps:
      1. Numerical (N): mean imputation fitted on train.
      2. Categorical (C): OrdinalEncoder fitted on train (unknown → -1).
      3. Concatenate [N_imputed | C_encoded] → float32 array.
    """

    def fit(
        self,
        N_train: Optional[np.ndarray],
        C_train: Optional[np.ndarray],
    ) -> "_TalentDataConverter":
        if N_train is not None:
            self._num_imp = SimpleImputer(strategy="mean")
            self._num_imp.fit(N_train.astype(np.float64))
        else:
            self._num_imp = None

        if C_train is not None:
            self._cat_enc = OrdinalEncoder(
                dtype=np.float32,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self._cat_enc.fit(C_train.astype(str))
        else:
            self._cat_enc = None

        return self

    def transform(
        self,
        N: Optional[np.ndarray],
        C: Optional[np.ndarray],
    ) -> np.ndarray:
        parts: list[np.ndarray] = []

        if N is not None:
            if self._num_imp is not None:
                parts.append(
                    self._num_imp.transform(N.astype(np.float64)).astype(np.float32)
                )
            else:
                parts.append(N.astype(np.float32))

        if C is not None and self._cat_enc is not None:
            parts.append(self._cat_enc.transform(C.astype(str)))

        if not parts:
            raise ValueError(
                "Dataset has neither numerical nor categorical features."
            )
        return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# TALENT-compatible metrics
# ---------------------------------------------------------------------------

def _ensure_proba(arr: np.ndarray) -> np.ndarray:
    """Convert logits or raw scores to a proper probability matrix."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return np.stack([1.0 - arr, arr], axis=1)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.allclose(row_sums, 1.0, atol=1e-3):
        # Already probabilities — re-normalize in float64 so rows sum to
        # exactly 1 (float32→float64 conversion can leave a small residual
        # that triggers sklearn's brier_score / log_loss warnings).
        arr = arr / row_sums
    else:
        # Raw logits — apply softmax.
        arr = arr - arr.max(axis=1, keepdims=True)
        e = np.exp(arr)
        arr = e / e.sum(axis=1, keepdims=True)
    return arr


def _build_y_info(y_train: np.ndarray, task_type: str) -> dict:
    """Build the y_info dict that TALENT's metric() function expects."""
    if task_type == "regression":
        mean = float(y_train.mean())
        std = float(y_train.std())
        if std == 0.0:
            std = 1.0
        return {"policy": "mean_std", "mean": mean, "std": std}
    classes = np.unique(y_train.astype(np.int64))
    return {"policy": "none", "classes": classes, "n_classes": len(classes)}


def _compute_metrics(
    proba: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    y_info: dict,
    threshold: Optional[float] = None,
) -> Tuple[tuple, tuple]:
    """Compute TALENT-compatible metrics. Matches base.Method.metric() exactly.

    Returns (metric_values, metric_names).
    """
    if task_type == "regression":
        preds = np.asarray(proba).ravel()
        mae  = skm.mean_absolute_error(y_test, preds)
        rmse = skm.mean_squared_error(y_test, preds) ** 0.5
        r2   = skm.r2_score(y_test, preds)
        # Note: we pass raw y (not z-normalized), so no std rescaling needed.
        return (mae, r2, rmse), ("MAE", "R2", "RMSE")

    proba = _ensure_proba(proba)

    if task_type == "binclass":
        hard = (
            (proba[:, 1] >= threshold).astype(int)
            if threshold is not None
            else proba.argmax(axis=1)
        )
        accuracy   = skm.accuracy_score(y_test, hard)
        avg_recall = skm.balanced_accuracy_score(y_test, hard)
        avg_prec   = skm.precision_score(y_test, hard, average="macro", zero_division=0)
        f1         = skm.f1_score(y_test, hard, average="binary")
        log_loss   = skm.log_loss(y_test, proba, labels=y_info["classes"])
        auc = (
            skm.roc_auc_score(y_test, proba[:, 1], labels=y_info["classes"])
            if len(np.unique(y_test)) == 2
            else float("nan")
        )
        from TALENT.model.lib.calibration import brier_score, expected_calibration_error
        brier = brier_score(y_test, proba)
        ece   = expected_calibration_error(y_test, proba)
        return (
            (accuracy, avg_recall, avg_prec, f1, log_loss, auc, brier, ece),
            ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC", "Brier", "ECE"),
        )

    # multiclass
    hard       = proba.argmax(axis=1)
    accuracy   = skm.accuracy_score(y_test, hard)
    avg_recall = skm.balanced_accuracy_score(y_test, hard)
    avg_prec   = skm.precision_score(y_test, hard, average="macro", zero_division=0)
    f1         = skm.f1_score(y_test, hard, average="macro")
    log_loss   = skm.log_loss(y_test, proba, labels=y_info["classes"])
    try:
        auc = skm.roc_auc_score(
            y_test, proba, multi_class="ovo", average="macro", labels=y_info["classes"]
        )
    except Exception:
        auc = float("nan")
    from TALENT.model.lib.calibration import brier_score, expected_calibration_error
    brier = brier_score(y_test, proba)
    ece   = expected_calibration_error(y_test, proba)
    return (
        (accuracy, avg_recall, avg_prec, f1, log_loss, auc, brier, ece),
        ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC", "Brier", "ECE"),
    )


def _tune_threshold(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    """Tune binary decision threshold on the validation set (maximise F1)."""
    try:
        from TALENT.model.lib.threshold import tune_threshold
        t, _ = tune_threshold(y_val, proba_val, metric="f1")
        return float(t)
    except Exception:
        # Fallback: simple grid search
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            hard  = (proba_val[:, 1] >= t).astype(int)
            score = skm.f1_score(y_val, hard, average="binary", zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        return best_t


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ricotta(
    cfg: Dict[str, Any],
    train_val_data: Tuple,
    test_data: Tuple,
    info: Dict[str, Any],
    seed_num: int = 1,
    save_path: str = "",
) -> Any:
    """Fit LoopICL on a TALENT dataset and return a RunResult.

    Parameters
    ----------
    cfg : dict
        One entry from a ricotta_configs.py CONFIGS list.
    train_val_data, test_data, info
        As returned by ``TALENT.model.lib.data.get_dataset``.
    seed_num : int
        Number of independent seeds to run and average over.
    save_path : str
        Kept for API compatibility (LoopICL has no model checkpoints to save).

    Returns
    -------
    TALENT.api.RunResult
    """
    from TALENT.api import RunResult

    task_type     = info.get("task_type", "binclass")
    is_regression = task_type == "regression"
    cfg_supports_regression = cfg.get("supports_regression", False)

    # Regression datasets require supports_regression=True.
    if is_regression and not cfg_supports_regression:
        return None

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
        y_train = y_tv["train"].astype(np.float32)
        y_val   = y_tv["val"].astype(np.float32)
        y_test  = y_te["test"].astype(np.float32)
    else:
        # Labels may be strings (e.g. 'malignant', 'g', 'EI') — encode first.
        from sklearn.preprocessing import LabelEncoder
        _le = LabelEncoder()
        _le.fit(np.concatenate([
            y_tv["train"].ravel(), y_tv["val"].ravel(), y_te["test"].ravel()
        ]))
        y_train = _le.transform(y_tv["train"].ravel()).astype(np.int64)
        y_val   = _le.transform(y_tv["val"].ravel()).astype(np.int64)
        y_test  = _le.transform(y_te["test"].ravel()).astype(np.int64)

    # -----------------------------------------------------------------------
    # Feature conversion (fit on train only)
    # -----------------------------------------------------------------------
    converter  = _TalentDataConverter().fit(N_train, C_train)
    X_train    = converter.transform(N_train, C_train)
    X_val      = converter.transform(N_val,   C_val)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    X_test     = converter.transform(N_test,  C_test)

    y_info = _build_y_info(y_trainval, task_type)

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    # Pick the right checkpoint: regression uses reg_weights_path,
    # classification uses cls_weights_path (falling back to weights_path for
    # backward-compat with old single-path configs).
    if is_regression:
        weights_path = cfg["reg_weights_path"]
    else:
        weights_path = cfg.get("cls_weights_path") or cfg["weights_path"]
    model_type: str          = cfg.get("model", "LoopICL")
    n_estimators: int        = cfg.get("n_estimators", 8)
    batch_size: int          = cfg.get("batch_size", 8)
    num_loops                = cfg.get("num_loops", None)
    subsample_samples        = cfg.get("subsample_samples", None)
    max_features             = cfg.get("max_features", None)
    test_chunk_size          = cfg.get("test_chunk_size", None)
    loop_residual_scaling_fn      = cfg.get("loop_residual_scaling_fn", None)
    loop_residual_entropy_gamma   = cfg.get("loop_residual_entropy_gamma", None)
    delta_gate_tau                = cfg.get("delta_gate_tau", None)
    exit_q                        = cfg.get("exit_q", None)
    heuristic_exit_strategy       = cfg.get("heuristic_exit_strategy", None)
    heuristic_exit_threshold      = cfg.get("heuristic_exit_threshold", None)
    y_transform_methods      = cfg.get("y_transform_methods", None)
    device: str              = cfg.get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

        if model_type == "LoopICLEarlyExit":
            ricotta_model = _load_ricotta_early_exit_model(weights_path, device=device)
        elif model_type == "LoopICLEarlyExitV2":
            ricotta_model = _load_ricotta_early_exit_v2_model(weights_path, device=device)
        elif is_regression:
            ricotta_model = _load_ricotta_reg_model(weights_path, device=device)
        else:
            ricotta_model = _load_ricotta_model(weights_path, device=device)
        if loop_residual_scaling_fn is not None:
            ricotta_model.loop_residual_scaling_fn = loop_residual_scaling_fn
        if loop_residual_entropy_gamma is not None:
            ricotta_model.loop_residual_entropy_gamma = loop_residual_entropy_gamma
        if delta_gate_tau is not None:
            ricotta_model.delta_gate_tau = delta_gate_tau
        adapter = _RicottaAdapter(
            ricotta_model,
            num_loops=num_loops,
            test_chunk_size=test_chunk_size,
            exit_q=exit_q,
            heuristic_exit_strategy=heuristic_exit_strategy,
            heuristic_exit_threshold=heuristic_exit_threshold,
        )

        if is_regression:
            from loopicl.interface.wrap_model import LoopICLRegWrapper
            wrapper = LoopICLRegWrapper(adapter, device, dtype="bfloat16")
        else:
            from loopicl.interface.wrap_model import TabICLWrapper
            wrapper = TabICLWrapper(adapter, device, dtype="bfloat16")

        # ------------------------------------------------------------------
        # Build classifier / regressor
        # ------------------------------------------------------------------
        if is_regression:
            from loopicl.interface import LoopICLTabICLRegressor
            clf = LoopICLTabICLRegressor(
                wrapper=wrapper,
                n_estimators=n_estimators,
                batch_size=batch_size,
                subsample_samples=subsample_samples,
                max_features=max_features,
                random_state=seed,
                y_transform_methods=y_transform_methods,
            )
        else:
            from loopicl.interface import LoopICLClassifier
            clf = LoopICLClassifier(
                wrapper=wrapper,
                n_estimators=n_estimators,
                batch_size=batch_size,
                subsample_samples=subsample_samples,
                max_features=max_features,
                random_state=seed,
            )

        # ------------------------------------------------------------------
        # Binary: tune threshold on val using train-only context
        # ------------------------------------------------------------------
        threshold: Optional[float] = None
        if task_type == "binclass":
            clf.fit(X_train, y_train)
            proba_val  = clf.predict_proba(X_val)
            threshold  = _tune_threshold(proba_val, y_val.astype(np.int64))

        # ------------------------------------------------------------------
        # Fit on full train+val, predict on test
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        clf.fit(X_trainval, y_trainval)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        if is_regression:
            preds_raw  = clf.predict(X_test)
            proba_test = np.asarray(preds_raw).reshape(-1, 1)
        else:
            proba_test = clf.predict_proba(X_test)
        predict_time = time.perf_counter() - t1

        metrics, metric_names = _compute_metrics(
            proba_test, y_test.astype(np.int64 if not is_regression else np.float64),
            task_type, y_info, threshold,
        )

        per_seed.append({
            "seed":         seed,
            "loss":         float("nan"),
            "metrics":      tuple(float(x) for x in metrics),
            "metric_names": metric_names,
            "fit_time":     fit_time,
            "predict_time": predict_time,
            "proba":        proba_test,
            "threshold":    threshold,
            # RunResult shape compatibility
            "predictions":      proba_test,
            "predict_proba":    proba_test if not is_regression else None,
            "predict_labels":   (
                (proba_test[:, 1] >= threshold).astype(int)
                if task_type == "binclass" and threshold is not None
                else (proba_test.argmax(axis=1) if not is_regression else None)
            ),
        })
        fit_times.append(fit_time)
        predict_times.append(predict_time)

        # Free GPU memory before next seed
        del ricotta_model, adapter, wrapper, clf
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
        method_type="ricotta",
        config=cfg,
        save_path=save_path,
    )
