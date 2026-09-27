"""LoopICL AutoGluon AbstractModel wrapper for TabArena evaluation.

Loads a trained LoopICL checkpoint and exposes it through the LoopICLClassifier /
LoopICLTabICLRegressor sklearn interface so it slots directly into the TabArena
benchmark runner.

Usage in run_benchmark.py::

    from models.loopicl_model import LoopICLModel
    # add ("LoopICL", LoopICLModel, {"weights_path": "/path/to/step-XXXXX.ckpt"}) to _local
"""

from __future__ import annotations

import gc
import pandas as pd
import torch

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.common.utils.resource_utils import ResourceManager


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _load_loopicl_reg_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICLReg (quantile regression) checkpoint."""
    import os
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"LoopICLReg checkpoint not found: {weights_path!r}\n"
            "Update 'reg_weights_path' in your config file to point at a valid .ckpt file."
        )

    from loopicl.models.loopicl.model import LoopICLConfig
    from loopicl.models.loopicl.model_reg import LoopICLReg

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")
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


def _load_loopicl_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICL model from a training checkpoint.

    The checkpoint is the ``.ckpt`` file saved by the LoopICL trainer.
    It contains:
        - ``"config"``         : model config fields
        - ``"state_dict"``     : raw model weights
        - ``"ema_state_dict"`` : EMA weights (preferred when present)
    """
    import os
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"LoopICL checkpoint not found: {weights_path!r}\n"
            "Update 'weights_path' in your config file to point at a valid .ckpt file."
        )

    from loopicl.models.loopicl.model import LoopICL, LoopICLConfig

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])
    # Compat: older checkpoints use 'max_classes' instead of 'max_num_classes'
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")
    # Filter to known LoopICLConfig fields to avoid unexpected keyword errors
    known = set(LoopICLConfig.__dataclass_fields__)
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in known}
    model = LoopICL(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


def _load_loopicl_early_exit_v2_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a LoopICLEarlyExit checkpoint."""
    import os
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"LoopICLEarlyExit checkpoint not found: {weights_path!r}\n"
            "Update 'weights_path' in your config file to point at a valid .ckpt file."
        )

    from loopicl.models.loopicl.model_early_exit import LoopICLEarlyExit, LoopICLEarlyExitConfig

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")
    known = set(LoopICLEarlyExitConfig.__dataclass_fields__)
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in known}
    model = LoopICLEarlyExit(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Calling-convention adapter
# ---------------------------------------------------------------------------


class _LoopICLAdapter:
    """Adapts LoopICL.forward to the TabICLWrapper calling convention.

    TabICLWrapper calls ``model(X, y_train, return_logits=True)`` where:
        X       : (B, n_total, n_features)
        y_train : (B, n_train)         ← only train labels

    LoopICL.forward expects:
        X       : (B, n_total, n_features)
        y_train : (B, n_total)         ← all rows (test rows are zeroed)
        single_eval_pos : int

    This adapter pads y_train with zeros and calls forward accordingly.
    OOM is handled by calibrating a safe batch chunk size on the first call.
    """

    def __init__(
        self,
        loopicl: torch.nn.Module,
        num_loops: int | None = None,
        test_chunk_size: int | None = None,
        row_chunk_size: int | None = None,
        col_chunk_size: int | None = None,
        offload_x_brce: bool = False,
        exit_q: float | None = None,
    ) -> None:
        self._loopicl = loopicl
        self._num_loops = num_loops
        self._test_chunk_size = test_chunk_size
        self._row_chunk_size = row_chunk_size
        self._col_chunk_size = col_chunk_size
        self._offload_x_brce = offload_x_brce
        self._exit_q = exit_q
        self.max_classes = loopicl.cfg.max_num_classes
        # Persisted across calls: once a safe batch chunk size is found after
        # OOM, reuse it on subsequent calls instead of retrying from full size.
        self._safe_batch_chunk: int | None = None
        # LoopICLReg.forward returns (M, B, Q) — batch is dim 1.
        # LoopICL.forward returns (B, M, T) — batch is dim 0.
        from loopicl.models.loopicl.model_reg import LoopICLReg as _LoopICLReg
        self._output_batch_dim: int = 1 if isinstance(loopicl, _LoopICLReg) else 0

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
        if self._test_chunk_size is not None:
            kw["test_chunk_size"] = self._test_chunk_size
        if self._row_chunk_size is not None:
            kw["row_chunk_size"] = self._row_chunk_size
        if self._col_chunk_size is not None:
            kw["col_chunk_size"] = self._col_chunk_size
        if self._offload_x_brce:
            kw["offload_x_brce"] = True
        if self._exit_q is not None:
            kw["exit_q"] = self._exit_q

        return self._forward_oom_safe(X, y_full, kw)

    def _calibrate_chunk_size(self, X: torch.Tensor, y_full: torch.Tensor, kw: dict) -> int:
        """Run one estimator, measure memory delta, derive safe batch chunk size."""
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info(X.device.index or 0)
        try:
            with torch.no_grad():
                _ = self._loopicl(X[:1], y_full[:1], **kw)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "    [mem] OOM on single estimator — dataset likely too large. "
                "Set subsample_samples in your config to cap the training context."
            )
            return 1
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
            return self._loopicl(X, y_full, **kw)
        parts = []
        for i in range(0, X.shape[0], chunk_size):
            parts.append(self._loopicl(X[i:i + chunk_size], y_full[i:i + chunk_size], **kw))
        return torch.cat(parts, dim=self._output_batch_dim)

    def prefill(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """Build a LoopICLTrainCache for the prefill/decode path.

        Called by _wrapper_prefill when cache_context=True or test_chunk_size is set.
        X_train : (B, N_train, C)
        y_train : (B, N_train)  — train labels only, no zero-padding needed.
        """
        kw = {}
        if self._num_loops is not None:
            kw["num_loops"] = self._num_loops
        if self._col_chunk_size is not None:
            kw["col_chunk_size"] = self._col_chunk_size
        if self._offload_x_brce:
            kw["offload_x_brce"] = True
        return self._loopicl.prefill(X_train, y_train, **kw)

    def predict_cached(self, X_test: torch.Tensor, cache) -> torch.Tensor:
        """Predict test rows using a precomputed LoopICLTrainCache.

        Called by _wrapper_predict_cached for each test chunk.
        X_test : (B, M, C)
        Returns (B, M, T) logits.
        """
        return self._loopicl.predict_cached(X_test, cache)

    def eval(self) -> "_LoopICLAdapter":
        self._loopicl.eval()
        return self

    def train(self, mode: bool = True) -> "_LoopICLAdapter":
        self._loopicl.train(mode)
        return self

    def to(self, *args, **kwargs) -> "_LoopICLAdapter":
        self._loopicl.to(*args, **kwargs)
        return self

    def parameters(self):
        return self._loopicl.parameters()


# ---------------------------------------------------------------------------
# AbstractModel wrapper
# ---------------------------------------------------------------------------


class LoopICLModel(AbstractModel):
    """LoopICL in-context learning model for binary / multiclass classification.

    Hyperparameters (pass via ``model_hyperparameters`` or ``params``)
    ------------------------------------------------------------------
    weights_path : str
        Path to a trained LoopICL ``.ckpt`` checkpoint.
    n_estimators : int, optional
        Number of ensemble members (feature-shuffle repeats). Default 8.
    num_loops : int, optional
        Override the number of ICL outer-loop iterations at inference.
        If None, uses the value the model was trained with.
    batch_size : int, default 8
        Maximum number of ensemble members per forward call.
    subsample_samples : int, optional
        Cap the training context to this many randomly-sampled rows before
        each forward pass. None (default) means no limit.
    max_features : int, optional
        Cap the number of features per ensemble member. Each estimator
        independently draws a random subset of this size (TabFM style).
        None (default) uses all features.
    test_chunk_size : int, optional
        Process test rows in chunks of this size. When set, activates the
        prefill/decode path (train prefilled once, test decoded per chunk),
        dramatically reducing peak GPU memory for large test sets.
        None (default) runs a single concatenated forward pass.
    cache_context : bool, default False
        Build the KV cache at fit() time and reuse it across predict_proba()
        calls. Eliminates redundant train-side computation when predicting
        multiple times on the same fitted model.
    keep_cache_on_cpu : bool, default False
        Move each estimator's KV cache to CPU after prefill and restore it
        to GPU before decoding. Reduces peak VRAM at the cost of extra
        host↔device transfers. Useful when n_estimators × cache size
        exceeds available VRAM.
    quantize_kv_cache : bool, default False
        Int8-quantize the KV cache after prefill. Reduces cache VRAM ~4×
        at a small accuracy cost.
    offload_x_brce : bool, default False
        Keep X_BRCE on CPU and move chunks to GPU one at a time during each
        forward pass. Eliminates the X_BRCE GPU footprint (n_rows × n_features
        × embed_dim) at the cost of host↔device transfers. The primary fix for
        large datasets that OOM even with batch_size=1. Automatically activates
        _axial_step_chunked.
    test_chunk_size : int, optional
        Number of rows per GPU chunk in cross-col blocks and ICL (the row
        dimension). Also controls train-row chunking in cross-col when
        offload_x_brce=True. Default 512.
    row_chunk_size : int, optional
        Number of train rows per GPU chunk in cross-col blocks and the readout
        layer, used only when offload_x_brce=True. Defaults to test_chunk_size
        when not set.
    col_chunk_size : int, optional
        Number of columns per GPU chunk in within-col blocks (the column
        dimension), used only when offload_x_brce=True. Default C // 8
        (roughly keeps each chunk ~1 GB on GPU).
    """

    ag_name = "LoopICL"
    ag_key = "RICOTTA"
    _problem_types = {BINARY, MULTICLASS, REGRESSION}

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def _set_default_params(self) -> None:
        self._set_default_param_value("weights_path", None)
        self._set_default_param_value("reg_weights_path", None)
        self._set_default_param_value("supports_regression", False)
        self._set_default_param_value("y_transform_methods", None)
        self._set_default_param_value("n_estimators", None)
        self._set_default_param_value("num_loops", None)
        self._set_default_param_value("batch_size", 8)
        self._set_default_param_value("subsample_samples", None)
        self._set_default_param_value("max_features", None)
        self._set_default_param_value("test_chunk_size", None)
        self._set_default_param_value("cache_context", False)
        self._set_default_param_value("keep_cache_on_cpu", False)
        self._set_default_param_value("quantize_kv_cache", False)
        self._set_default_param_value("offload_x_brce", False)
        self._set_default_param_value("row_chunk_size", None)
        self._set_default_param_value("col_chunk_size", None)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ) -> None:
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 and is_available() else "cpu"
        is_regression = self.problem_type == REGRESSION

        n_estimators: int | None = self.params.get("n_estimators")
        num_loops: int | None = self.params.get("num_loops")
        batch_size: int = self.params.get("batch_size", 8)

        if is_regression:
            reg_weights_path: str | None = self.params.get("reg_weights_path")
            if not reg_weights_path:
                raise ValueError(
                    "LoopICLModel requires a 'reg_weights_path' hyperparameter for "
                    "regression tasks, pointing to a trained LoopICLReg checkpoint (.ckpt file)."
                )
            from loopicl.interface import LoopICLTabICLRegressor
            from loopicl.interface.wrap_model import LoopICLRegWrapper

            loopicl = _load_loopicl_reg_model(reg_weights_path, device=device)
            adapter = _LoopICLAdapter(
                loopicl,
                num_loops=num_loops,
                test_chunk_size=self.params.get("test_chunk_size"),
                row_chunk_size=self.params.get("row_chunk_size"),
                col_chunk_size=self.params.get("col_chunk_size"),
                offload_x_brce=self.params.get("offload_x_brce", False),
            )
            wrapper = LoopICLRegWrapper(adapter, device, dtype="bfloat16")
            clf = LoopICLTabICLRegressor(
                wrapper=wrapper,
                n_estimators=n_estimators if n_estimators is not None else 8,
                batch_size=batch_size,
                subsample_samples=self.params.get("subsample_samples"),
                max_features=self.params.get("max_features"),
                y_transform_methods=self.params.get("y_transform_methods"),
            )
        else:
            weights_path: str | None = self.params.get("weights_path")
            if not weights_path:
                raise ValueError(
                    "LoopICLModel requires a 'weights_path' hyperparameter pointing to "
                    "a trained LoopICL checkpoint (.ckpt file)."
                )
            from loopicl.interface import LoopICLClassifier
            from loopicl.interface.wrap_model import TabICLWrapper

            loopicl = _load_loopicl_model(weights_path, device=device)
            adapter = _LoopICLAdapter(
                loopicl,
                num_loops=num_loops,
                test_chunk_size=self.params.get("test_chunk_size"),
                row_chunk_size=self.params.get("row_chunk_size"),
                col_chunk_size=self.params.get("col_chunk_size"),
                offload_x_brce=self.params.get("offload_x_brce", False),
            )
            wrapper = TabICLWrapper(adapter, device, dtype="bfloat16")
            clf = LoopICLClassifier(
                wrapper=wrapper,
                n_estimators=n_estimators if n_estimators is not None else 8,
                batch_size=batch_size,
                subsample_samples=self.params.get("subsample_samples"),
                max_features=self.params.get("max_features"),
                test_chunk_size=self.params.get("test_chunk_size"),
                cache_context=self.params.get("cache_context", False),
                keep_cache_on_cpu=self.params.get("keep_cache_on_cpu", False),
                quantize_kv_cache=self.params.get("quantize_kv_cache", False),
            )

        X = self.preprocess(X, y=y)
        clf.fit(X=X, y=y)
        self.model = clf


# ---------------------------------------------------------------------------
# Early-exit v2 model wrapper
# ---------------------------------------------------------------------------


class LoopICLEarlyExitModel(AbstractModel):
    """LoopICLEarlyExit classification model for TabArena evaluation.

    Loads a checkpoint trained with EarlyExitTrainerV2 (gate-only stage,
    no loop residual scaling) and evaluates at a given exit quantile.

    Hyperparameters
    ---------------
    weights_path : str
        Path to the trained LoopICLEarlyExit ``.ckpt`` checkpoint.
    exit_q : float, default 0.5
        Quantile threshold for the learned exit gate.
        0.0 = exit as early as possible; 1.0 = run all loops (no early exit).
    n_estimators : int, default 8
    batch_size : int, default 8
    num_loops : int, optional
        Override the max loop count (default: value from checkpoint).
    subsample_samples : int, optional
    """

    ag_name = "LoopICLEarlyExit"
    ag_key = "RICOTTA_EE_V2"
    _problem_types = {BINARY, MULTICLASS}

    def _get_default_resources(self) -> tuple[int, int]:
        from autogluon.common.utils.resource_utils import ResourceManager
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict:
        return {"num_cpus": 1, "num_gpus": 1 if is_gpu_available else 0}

    def _set_default_params(self) -> None:
        self._set_default_param_value("weights_path", None)
        self._set_default_param_value("exit_q", 0.5)
        self._set_default_param_value("n_estimators", None)
        self._set_default_param_value("num_loops", None)
        self._set_default_param_value("batch_size", 8)
        self._set_default_param_value("subsample_samples", None)
        self._set_default_param_value("max_features", None)
        self._set_default_param_value("test_chunk_size", None)
        self._set_default_param_value("row_chunk_size", None)
        self._set_default_param_value("col_chunk_size", None)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ) -> None:
        from torch.cuda import is_available

        device = "cuda" if num_gpus != 0 and is_available() else "cpu"

        weights_path: str | None = self.params.get("weights_path")
        if not weights_path:
            raise ValueError(
                "LoopICLEarlyExitModel requires a 'weights_path' hyperparameter "
                "pointing to a trained LoopICLEarlyExit checkpoint (.ckpt file)."
            )

        from loopicl.interface import LoopICLClassifier
        from loopicl.interface.wrap_model import TabICLWrapper

        loopicl = _load_loopicl_early_exit_v2_model(weights_path, device=device)
        adapter = _LoopICLAdapter(
            loopicl,
            num_loops=self.params.get("num_loops"),
            test_chunk_size=self.params.get("test_chunk_size"),
            row_chunk_size=self.params.get("row_chunk_size"),
            col_chunk_size=self.params.get("col_chunk_size"),
            exit_q=self.params.get("exit_q"),
        )
        wrapper = TabICLWrapper(adapter, device, dtype="bfloat16")
        n_estimators = self.params.get("n_estimators")
        clf = LoopICLClassifier(
            wrapper=wrapper,
            n_estimators=n_estimators if n_estimators is not None else 8,
            batch_size=self.params.get("batch_size", 8),
            subsample_samples=self.params.get("subsample_samples"),
            max_features=self.params.get("max_features"),
            test_chunk_size=self.params.get("test_chunk_size"),
        )

        X = self.preprocess(X, y=y)
        clf.fit(X=X, y=y)
        self.model = clf
