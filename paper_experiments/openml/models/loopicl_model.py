"""LoopICL thin wrapper for OpenML benchmark (no autogluon).

Loads a trained LoopICL checkpoint and exposes it through the IrisClassifier
sklearn interface, using the same fit/predict_proba/predict protocol as other openml models.

Usage in run_benchmark.py configs::

    # configs/loopicl_c1.py
    CONFIG = {"name": "LoopICL-c1", "model": "LoopICL", "weights_path": "/path/to/step.ckpt"}
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _load_loopicl_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    # Load to CPU first so we never hold both the checkpoint tensors and the
    # moved model simultaneously on GPU (which would temporarily use 2× VRAM).
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_config_dict = dict(checkpoint["config"])
    if "max_classes" in model_config_dict and "max_num_classes" not in model_config_dict:
        model_config_dict["max_num_classes"] = model_config_dict.pop("max_classes")

    from loopicl.models.loopicl.model import LoopICL

    model = LoopICL(**model_config_dict)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint  # free CPU memory before moving model to GPU
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Regression checkpoint loader
# ---------------------------------------------------------------------------


def _load_loopicl_reg_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model_config_dict = dict(checkpoint["config"])
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


class _LoopICLForTabICL:
    """Adapts LoopICL to the calling convention expected by IrisClassifier."""

    def __init__(self, loopicl: torch.nn.Module, max_train_samples: int | None = None):
        self._loopicl = loopicl
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

        loopicl_kwargs: dict = {"single_eval_pos": n_train}
        if "num_loops" in kwargs:
            loopicl_kwargs["num_loops"] = kwargs["num_loops"]

        return self._forward_with_oom_retry(X, y_full, loopicl_kwargs)

    def _forward_with_oom_retry(self, X, y_full, loopicl_kwargs):
        device = X.device
        chunk_size = X.shape[0]

        while True:
            try:
                if X.device.type == device.type:
                    return self._loopicl(X, y_full, **loopicl_kwargs)
                else:
                    results = []
                    for start in range(0, X.shape[0], chunk_size):
                        end = min(start + chunk_size, X.shape[0])
                        out = self._loopicl(
                            X[start:end].to(device),
                            y_full[start:end].to(device),
                            **loopicl_kwargs,
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
        self._loopicl.eval()
        return self

    def train(self, mode: bool = True):
        self._loopicl.train(mode)
        return self

    def to(self, *args, **kwargs):
        self._loopicl.to(*args, **kwargs)
        return self

    def parameters(self):
        return self._loopicl.parameters()


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LoopICLModel:
    """LoopICL in-context learning model for binary / multiclass classification.

    Hyperparameters (pass via config)
    ----------------------------------
    weights_path : str
        Path to a trained LoopICL ``.ckpt`` checkpoint.
    n_estimators : int, optional
    num_loops : int, optional
    batch_size : int, default 8
    subsample_samples : int, optional
    capture_hidden_states : bool, default False
        If True, ``compute_hs_metrics(X_train, y_train, X_test)`` becomes
        available after ``fit()``.  Results are stored in ``self.hs_metrics``
        (dict of np.ndarray) and ``self.loop_logits`` (list of np.ndarray).
    capture_embeddings : bool, default False
        If True, raw row-stream hidden states are stored in
        ``self.loop_embeddings`` (shape ``(n_loops+1, n_points, D)``).
        Requires ``capture_hidden_states=True``.
    """

    name = "LoopICL"

    def __init__(
        self,
        weights_path: str | None = None,
        weights_path_reg: str | None = None,
        n_estimators: int | None = None,
        num_loops: int | None = None,
        batch_size: int = 8,
        test_chunk_size: int | None = None,
        subsample_samples: int | None = None,
        max_features: int | None = None,
        multiclass_strategy: str = "tabicl",
        capture_hidden_states: bool = False,
        capture_embeddings: bool = False,
        capture_test_embeddings: bool = False,
        capture_layer_metrics: bool = False,
        capture_layer_matrix: bool = False,
        layer_matrix_max_samples: int = 200,
        layers_as_loops: bool = False,
        capture_lyapunov: bool = False,
        lyapunov_n_exponents: int = 5,
        lyapunov_per_block: bool = False,
        capture_kv_rank: bool = False,
        capture_kv_compression: bool = False,
        kv_compress_ranks: list[int] | None = None,
        n_timing_repeats: int = 3,
        loop_residual_n_for_scaling: int | None = None,
        loop_residual_scaling_fn: str | None = None,
        loop_residual_scaling_n_ref: int | None = None,
        capture_quantization: bool = False,
        subsample_train: int | None = None,
        subsample_test: int | None = None,
        norm_methods: list[str] | str | None = None,
        save_train_data: bool = False,
        capture_attention_heads: bool = False,
        capture_head_ablation: bool = False,
        capture_loop_predictions: bool = False,
    ):
        self._weights_path = weights_path
        self._weights_path_reg = weights_path_reg
        self._n_estimators = n_estimators
        self._num_loops = num_loops
        self._batch_size = batch_size
        self._test_chunk_size = test_chunk_size
        self._subsample_samples = subsample_samples
        self._max_features = max_features
        self._multiclass_strategy = multiclass_strategy
        self.capture_hidden_states = capture_hidden_states or capture_test_embeddings or layers_as_loops
        self.capture_embeddings = capture_embeddings
        self.capture_test_embeddings = capture_test_embeddings
        self.capture_layer_metrics = capture_layer_metrics
        self.capture_layer_matrix = capture_layer_matrix
        self._layer_matrix_max_samples = layer_matrix_max_samples
        self.layers_as_loops = layers_as_loops
        self.capture_lyapunov = capture_lyapunov
        self._lyapunov_n_exponents = lyapunov_n_exponents
        self._lyapunov_per_block = lyapunov_per_block
        self.capture_kv_rank = capture_kv_rank
        self.capture_kv_compression = capture_kv_compression
        self._kv_compress_ranks = kv_compress_ranks if kv_compress_ranks is not None else [1, 2, 4, 8, 16, 32]
        self._n_timing_repeats = n_timing_repeats
        self._loop_residual_n_for_scaling = loop_residual_n_for_scaling
        self._loop_residual_scaling_fn = loop_residual_scaling_fn
        self._loop_residual_scaling_n_ref = loop_residual_scaling_n_ref
        self.capture_quantization = capture_quantization
        self._subsample_train = subsample_train
        self._subsample_test = subsample_test
        self._norm_methods = norm_methods
        self.save_train_data = save_train_data
        self.capture_attention_heads = capture_attention_heads
        self.capture_head_ablation = capture_head_ablation
        self.capture_loop_predictions = capture_loop_predictions

        # populated by compute_hs_metrics() / compute_loop_predictions() / compute_kv_rank_metrics() / compute_attention_head_metrics()
        self.hs_metrics: dict | None = None
        self.loop_logits: list[np.ndarray] | None = None
        self.kv_rank_metrics: dict | None = None
        self.kv_compression_metrics: dict | None = None
        self.quantization_metrics: dict | None = None
        self.attention_head_metrics: dict | None = None

    def _load_weights(self, weights_path: str, device: str) -> torch.nn.Module:
        return _load_loopicl_model(weights_path, device=device)

    def fit(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._problem_type = problem_type
        self._device = device

        if problem_type == "regression":
            reg_path = self._weights_path_reg or self._weights_path
            if not reg_path:
                raise ValueError(
                    "LoopICLModel requires 'weights_path_reg' (or 'weights_path') for regression."
                )
            from loopicl.interface import LoopICLTabICLRegressor
            from loopicl.interface.wrap_model import LoopICLRegWrapper

            loopicl = _load_loopicl_reg_model(reg_path, device=device)
            self._loopicl = loopicl
            adapter = _LoopICLForTabICL(loopicl)
            wrapper = LoopICLRegWrapper(adapter, device, dtype="bfloat16", num_loops=self._num_loops)
            clf = LoopICLTabICLRegressor(
                wrapper=wrapper,
                n_estimators=self._n_estimators if self._n_estimators is not None else 8,
                batch_size=self._batch_size,
                subsample_samples=self._subsample_samples,
                max_features=self._max_features,
            )
        else:
            if not self._weights_path:
                raise ValueError(
                    "LoopICLModel requires a 'weights_path' pointing to a trained LoopICL checkpoint."
                )
            from loopicl.interface import LoopICLClassifier
            from loopicl.interface.wrap_model import TabICLWrapper

            loopicl = self._load_weights(self._weights_path, device=device)
            self._loopicl = loopicl  # kept for compute_hs_metrics()

            if self._loop_residual_n_for_scaling is not None and loopicl.use_loop_residual_scaling:
                actual_n = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops
                loopicl.loop_residual_lambda *= actual_n / self._loop_residual_n_for_scaling
            if self._loop_residual_scaling_fn is not None:
                loopicl.loop_residual_scaling_fn = self._loop_residual_scaling_fn
            if self._loop_residual_scaling_n_ref is not None:
                loopicl.loop_residual_scaling_n_ref = self._loop_residual_scaling_n_ref

            adapter = _LoopICLForTabICL(loopicl)
            wrapper = TabICLWrapper(adapter, device, dtype="bfloat16", num_loops=self._num_loops)
            clf = LoopICLClassifier(
                wrapper=wrapper,
                n_estimators=self._n_estimators if self._n_estimators is not None else 8,
                batch_size=self._batch_size,
                test_chunk_size=self._test_chunk_size,
                subsample_samples=self._subsample_samples,
                max_features=self._max_features,
                multiclass_strategy=self._multiclass_strategy,
                norm_methods=self._norm_methods,
            )

        X = X.copy(); X.columns = X.columns.astype(str)
        clf.fit(X=X, y=y)
        self._model = clf
        self._y_train = y  # kept for label encoding in compute_hs_metrics()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy(); X.columns = X.columns.astype(str)
        return self._model.predict(X)

    def compute_loop_predictions(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """Lightweight forward pass that captures per-loop probabilities only.

        Much faster than ``compute_hs_metrics`` — no hidden-state metrics,
        no CKA, no layer matrices.  Populates ``self.loop_logits`` with one
        probability array per loop step (index 0 = pre-loop init).

        Results
        -------
        self.loop_logits : list[np.ndarray]
            Per-loop prediction probabilities for test rows,
            each shape (n_test, n_classes).  Index 0 = pre-loop init.
        self.hs_metrics  : None  (not computed)
        """
        import torch
        from models.loopicl_hs_forward import forward_loop_predictions_only

        clf     = self._model
        device  = self._device
        loopicl = self._loopicl

        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)
        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)

        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        X_tr  = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te  = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)

        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        num_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops

        with torch.no_grad():
            _, all_loop_logits = forward_loop_predictions_only(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
                test_chunk_size=self._test_chunk_size,
            )

        T      = clf.softmax_temperature
        n_cls  = clf.n_classes_

        def _to_proba(logits_tensor: "torch.Tensor") -> np.ndarray:
            out = logits_tensor[0, :, :n_cls].float().cpu().numpy()
            out = out / max(T, 1e-8)
            out = out - out.max(axis=-1, keepdims=True)
            exp_out = np.exp(out)
            proba = exp_out / exp_out.sum(axis=-1, keepdims=True)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba

        self.hs_metrics  = None
        self.loop_logits = [_to_proba(lg) for lg in all_loop_logits]
        self.loop_embeddings = None

    def compute_hs_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        capture_embeddings: bool = False,
        capture_test_embeddings: bool = False,
    ) -> None:
        """Run a single forward pass through LoopICL and store hidden-state
        metrics + per-loop logits in ``self.hs_metrics`` / ``self.loop_logits``.

        Bypasses IrisClassifier ensembling (B=1, no class/feature shuffle) but
        reuses its fitted preprocessors (XEncoder, LabelEncoder) and applies
        the same softmax temperature + re-normalisation so that loop_logits are
        directly comparable to predict_proba output.

        Results
        -------
        self.hs_metrics   : dict[str, np.ndarray]
            Per-point arrays shape (n_loops, n_train) or (n_loops, n_test),
            plus aggregate "_mean" arrays of shape (n_loops,).
        self.loop_logits  : list[np.ndarray]
            Per-loop prediction probabilities for test rows,
            each shape (n_test, n_classes).
        self.loop_embeddings : dict{"train_row", "test_row"} | None
            Raw row-stream hidden states, shape (n_loops+1, n_points, D).
            Index 0 = pre-loop init.  Only set when capture_embeddings=True.
        """
        import torch
        from models.loopicl_hs_forward import forward_with_hs_metrics

        clf    = self._model   # fitted IrisClassifier
        device = self._device
        loopicl = self._loopicl

        # ── Feature preprocessing (same XEncoder fitted during fit()) ──────
        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)
        X_tr_enc = clf.X_encoder_.transform(X_train)   # (n_train, n_features)
        X_te_enc = clf.X_encoder_.transform(X_test)    # (n_test,  n_features)

        # ── Label encoding ───────────────────────────────────────────────────
        is_regression = (self._problem_type == "regression")
        if is_regression:
            y_tr_enc = y_train.values.astype(np.float32)  # raw float targets
        else:
            y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)  # (n_train,)

        # ── Optional subsampling (same limit as IrisClassifier) ─────────────
        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        # ── Build tensors with batch dim = 1 ────────────────────────────────
        X_tr = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)   # (1, n_total, n_features)

        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_dtype = torch.float32 if is_regression else torch.long
        y_tr   = torch.tensor(y_tr_enc, dtype=y_dtype, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=y_dtype, device=device)
        y_full[:, :n_train] = y_tr

        num_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops

        y_int_train = None if is_regression else y_tr_enc.astype(np.int64)

        with torch.no_grad():
            _, all_loop_logits, metrics, embeddings = forward_with_hs_metrics(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
                y_int_train=y_int_train,
                capture_embeddings=capture_embeddings,
                capture_test_embeddings=capture_test_embeddings,
                capture_layer_metrics=self.capture_layer_metrics,
                capture_layer_matrix=self.capture_layer_matrix,
                layer_matrix_max_samples=self._layer_matrix_max_samples,
                layers_as_loops=self.layers_as_loops,
                capture_lyapunov=self.capture_lyapunov,
                lyapunov_n_exponents=self._lyapunov_n_exponents,
                lyapunov_per_block=self._lyapunov_per_block,
            )

        # ── Convert logits → output arrays ───────────────────────────────────
        if is_regression:
            # Store raw quantile predictions per loop: (n_test, num_quantiles)
            loop_probas: list[np.ndarray] = [
                lg[0].float().cpu().numpy() for lg in all_loop_logits
            ]
        else:
            T     = clf.softmax_temperature
            n_cls = clf.n_classes_

            def _to_proba(logits_tensor: "torch.Tensor") -> np.ndarray:
                # logits_tensor: (B, n_test, max_classes); B=1, slice to actual n_classes
                out = logits_tensor[0, :, :n_cls].float().cpu().numpy()  # (n_test, n_classes)
                # temperature softmax
                out = out / max(T, 1e-8)
                out = out - out.max(axis=-1, keepdims=True)
                exp_out = np.exp(out)
                proba = exp_out / exp_out.sum(axis=-1, keepdims=True)
                # re-normalise (matches IrisClassifier final step)
                proba = proba / proba.sum(axis=1, keepdims=True)
                return proba

            loop_probas = [_to_proba(lg) for lg in all_loop_logits]

        self.hs_metrics = metrics
        self.loop_logits = loop_probas
        self.loop_embeddings = embeddings

    def compute_kv_rank_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """Run a forward pass and capture KV low-rank statistics per loop.

        Results are stored in ``self.kv_rank_metrics`` (dict of np.ndarray /
        float).  Key headline metrics:

        - ``K_temporal_stable_rank_mean``  scalar — avg stable rank of K
          across loops (1 = perfectly rank-1, n_loops = full rank)
        - ``K_temporal_evr_mean``          (n_loops,) — avg cumulative
          explained variance ratio across blocks
        - ``K_temporal_singular_values_norm``  (n_blocks, n_loops) — normalized
          singular values; rapid decay → low rank
        - ``K_delta_stable_rank_mean``     stable rank of *changes* in K
        - ``K_spatial_stable_rank_mean``   (n_loops,) — spatial rank of K
          at each loop iteration
        """
        import torch
        from models.loopicl_kv_rank import forward_with_kv_rank_metrics

        clf     = self._model
        device  = self._device
        loopicl = self._loopicl

        import time as _time
        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)

        _t = _time.time()
        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)
        print(f"        [kv] encode:        {_time.time()-_t:.3f}s  n_train={len(X_tr_enc)} n_test={len(X_te_enc)}")

        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        X_tr = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        # KV hooks only capture training tokens (x[:, :N_train]) — 1 test token suffices,
        # avoiding the O((N_train + N_test)^2) attention cost on GPU.
        X_te = torch.tensor(X_te_enc[:1], dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)

        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]
        print(f"        [kv] tensor shape:  n_train={n_train} n_total={n_total} (1 test token for kv pass)")

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        num_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops
        print(f"        [kv] num_loops={num_loops}")

        _t = _time.time()
        with torch.no_grad():
            _, kv_metrics = forward_with_kv_rank_metrics(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
            )
        print(f"        [kv] forward+hooks: {_time.time()-_t:.3f}s")

        self.kv_rank_metrics = kv_metrics


    def compute_kv_compression_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
        n_timing_repeats: int = 3,
    ) -> None:
        """Run inference at each rank in ``kv_compress_ranks`` with SVD-truncated
        K and V, and record the resulting score and wall-clock time vs the
        full-rank baseline.

        Results stored in ``self.kv_compression_metrics``:
          {
            "metric":              "roc_auc" | "log_loss" | ...,
            "full":                score with no compression,
            "full_time_ms":        median wall time for full-rank forward (ms),
            "rank_1":              score with rank-1 K/V,
            "rank_1_time_ms":      median wall time for rank-1 forward (ms),
            ...
            "prefill_time_ms":     median time for prefill() (None if unavailable),
            "cached_time_ms":      median time for predict_cached() per call (None if unavailable),
          }

        Timing uses ``n_timing_repeats`` warm runs after one throwaway warm-up
        and reports the median.  GPU timings synchronize before/after each run.
        """
        import time
        import torch
        from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
        from models.loopicl_kv_rank import (
            forward_with_kv_compression,
            forward_with_kv_reuse,
            forward_with_kv_mean_reuse,
            forward_with_kv_last_reuse,
            forward_with_kv_dim_split_reuse,
        )

        clf     = self._model
        loopicl = self._loopicl
        device  = self._device
        is_cuda = (device != "cpu") and torch.cuda.is_available()

        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)

        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)
        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        X_tr  = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te  = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)   # (1, n_train + n_test, C)
        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        n_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops
        n_cls   = clf.n_classes_
        T       = getattr(clf, "softmax_temperature", 1.0) or 1.0

        def _sync():
            if is_cuda:
                torch.cuda.synchronize()

        def _timed(fn) -> tuple:
            """Warm-up once, then time ``fn()`` for ``n_timing_repeats`` runs.
            Returns (last_result, median_ms)."""
            with torch.no_grad():
                result = fn()
            _sync()
            times = []
            for _ in range(n_timing_repeats):
                _sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    result = fn()
                _sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            return result, float(np.median(times))

        def _logits_to_proba(logits: torch.Tensor) -> np.ndarray:
            out = logits[0, :, :n_cls].float().cpu().numpy()
            out = out / max(float(T), 1e-8)
            out -= out.max(axis=-1, keepdims=True)
            exp_out = np.exp(out)
            proba = exp_out / exp_out.sum(axis=-1, keepdims=True)
            return proba / proba.sum(axis=1, keepdims=True)

        def _score(proba: np.ndarray) -> tuple[str, float]:
            y_true = y_test.to_numpy()
            if problem_type == "binary":
                return "roc_auc", float(roc_auc_score(y_true, proba[:, 1]))
            elif problem_type == "multiclass":
                return "log_loss", float(-log_loss(y_true, proba))
            else:
                return "neg_mse", float(-mean_squared_error(y_true, proba[:, 0]))

        results: dict = {}

        # ---- Full-rank baseline (score + time) ----
        logits, full_ms = _timed(
            lambda: loopicl(X_all, y_full, single_eval_pos=n_train, num_loops=n_loops)
        )
        metric_name, score = _score(_logits_to_proba(logits))
        results["metric"]       = metric_name
        results["full"]         = score
        results["full_time_ms"] = full_ms
        print(f"        [kv_compress] rank=full  {metric_name}={score:.4f}  "
              f"time={full_ms:.1f}ms")

        # ---- KV reuse (loop-1 / block-0 frozen for all subsequent passes) ----
        logits_reuse, reuse_ms = _timed(
            lambda: forward_with_kv_reuse(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=n_loops,
            )
        )
        _, score_reuse = _score(_logits_to_proba(logits_reuse))
        results["kv_reuse"]         = score_reuse
        results["kv_reuse_time_ms"] = reuse_ms
        print(f"        [kv_compress] kv_reuse   {metric_name}={score_reuse:.4f}  "
              f"drop={results['full'] - score_reuse:+.4f}  time={reuse_ms:.1f}ms")

        # ---- KV mean reuse (mean K/V across all loops/blocks frozen) ----
        logits_mean, mean_ms = _timed(
            lambda: forward_with_kv_mean_reuse(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=n_loops,
            )
        )
        _, score_mean = _score(_logits_to_proba(logits_mean))
        results["kv_mean_reuse"]         = score_mean
        results["kv_mean_reuse_time_ms"] = mean_ms
        print(f"        [kv_compress] kv_mean    {metric_name}={score_mean:.4f}  "
              f"drop={results['full'] - score_mean:+.4f}  time={mean_ms:.1f}ms")

        # ---- KV last reuse (last-loop K/V frozen for all loops/blocks) ----
        logits_last, last_ms = _timed(
            lambda: forward_with_kv_last_reuse(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=n_loops,
            )
        )
        _, score_last = _score(_logits_to_proba(logits_last))
        results["kv_last_reuse"]         = score_last
        results["kv_last_reuse_time_ms"] = last_ms
        print(f"        [kv_compress] kv_last    {metric_name}={score_last:.4f}  "
              f"drop={results['full'] - score_last:+.4f}  time={last_ms:.1f}ms")

        # ---- KV dim-split reuse (stable dims frozen after loop 1) ----
        for stable_frac in (0.25, 0.50, 0.75):
            logits_ds, ds_ms = _timed(
                lambda f=stable_frac: forward_with_kv_dim_split_reuse(
                    loopicl, X_all, y_full,
                    single_eval_pos=n_train,
                    stable_frac=f,
                    num_loops=n_loops,
                )
            )
            _, score_ds = _score(_logits_to_proba(logits_ds))
            key = f"kv_dim_split_{int(stable_frac*100)}"
            results[key]              = score_ds
            results[f"{key}_time_ms"] = ds_ms
            print(f"        [kv_compress] {key} {metric_name}={score_ds:.4f}  "
                  f"drop={results['full'] - score_ds:+.4f}  time={ds_ms:.1f}ms")

        # ---- Compressed ranks (score + time) ----
        for rank_k in self._kv_compress_ranks:
            logits_k, rank_ms = _timed(
                lambda r=rank_k: forward_with_kv_compression(
                    loopicl, X_all, y_full,
                    single_eval_pos=n_train,
                    rank_k=r,
                    num_loops=n_loops,
                )
            )
            _, score_k = _score(_logits_to_proba(logits_k))
            results[f"rank_{rank_k}"]         = score_k
            results[f"rank_{rank_k}_time_ms"] = rank_ms
            print(f"        [kv_compress] rank={rank_k:<4d}  {metric_name}={score_k:.4f}  "
                  f"drop={results['full'] - score_k:+.4f}  time={rank_ms:.1f}ms")

        # ---- Prefill + predict_cached (when supported by this model variant) ----
        if hasattr(loopicl, "prefill") and hasattr(loopicl, "predict_cached"):
            from models.loopicl_kv_rank import _KVCompressionHooks

            # Full rank: one-time train cost + per-query cached cost
            _, prefill_ms = _timed(lambda: loopicl.prefill(X_tr, y_tr, num_loops=n_loops))
            with torch.no_grad():
                cache = loopicl.prefill(X_tr, y_tr, num_loops=n_loops)
            _, cached_ms = _timed(lambda: loopicl.predict_cached(X_te, cache))
            results["prefill_time_ms"] = prefill_ms
            results["cached_time_ms"]  = cached_ms
            speedup = full_ms / cached_ms if cached_ms > 0 else float("inf")
            print(f"        [kv_compress] prefill=full  "
                  f"prefill={prefill_ms:.1f}ms  cached={cached_ms:.1f}ms  "
                  f"speedup={speedup:.1f}x vs full")

            # Compressed ranks: prefill with rank-k KVs + predict_cached with them
            for rank_k in self._kv_compress_ranks:
                # Time prefill with rank-k hooks active (KVs compressed at capture time)
                def _prefill_compressed(r=rank_k):
                    h = _KVCompressionHooks(r)
                    h.register(loopicl)
                    try:
                        return loopicl.prefill(X_tr, y_tr, num_loops=n_loops)
                    finally:
                        h.remove()

                _, p_ms = _timed(_prefill_compressed)

                # Build a fresh compressed cache then time predict_cached
                hooks_k = _KVCompressionHooks(rank_k)
                hooks_k.register(loopicl)
                try:
                    with torch.no_grad():
                        cache_k = loopicl.prefill(X_tr, y_tr, num_loops=n_loops)
                finally:
                    hooks_k.remove()

                _, c_ms = _timed(lambda c=cache_k: loopicl.predict_cached(X_te, c))
                results[f"prefill_rank_{rank_k}_time_ms"] = p_ms
                results[f"cached_rank_{rank_k}_time_ms"]  = c_ms
                speedup_k = full_ms / c_ms if c_ms > 0 else float("inf")
                print(f"        [kv_compress] prefill=rank_{rank_k:<4d}  "
                      f"prefill={p_ms:.1f}ms  cached={c_ms:.1f}ms  "
                      f"speedup={speedup_k:.1f}x vs full")
        else:
            results["prefill_time_ms"] = None
            results["cached_time_ms"]  = None
            for rank_k in self._kv_compress_ranks:
                results[f"prefill_rank_{rank_k}_time_ms"] = None
                results[f"cached_rank_{rank_k}_time_ms"]  = None

        self.kv_compression_metrics = results

    def compute_quantization_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
        n_timing_repeats: int = 5,
    ) -> None:
        """Compare baseline (bfloat16), weight-only INT8, and compute INT8 inference.

        On A100, compute INT8 uses Tensor Cores at 2× FP16 throughput.
        Requires: pip install torchao

        Results stored in ``self.quantization_metrics``.  Per mode ∈
        {baseline, weight_int8, weight_int4, compute_int8, compute_int4}:
          {
            "metric":                    str,
            "<mode>_score":              float,
            "<mode>_time_ms":            float,  # median wall time over n_timing_repeats
            "<mode>_peak_mb":            float,  # VRAM high-watermark during timed runs
                                                 # (weights + activations + shared tensors)
            "<mode>_params_mb":          float,  # static parameter storage (sum of p.nbytes())
                                                 # directly reflects quantized dtype sizes
            "<mode>_activation_mb":      float,  # peak_mb − params_mb ≈ ephemeral overhead
          }
        """
        import copy
        import time
        import gc
        import torch
        from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

        clf     = self._model
        loopicl = self._loopicl
        device  = self._device
        is_cuda = (device != "cpu") and torch.cuda.is_available()

        # ── Preprocess (same pipeline as kv_compression) ───────────────────
        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)

        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)
        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        # Cap train + test to prevent OOM. Raw model forward is O((n_train + n_test)²)
        # in attention; without IrisClassifier's built-in subsampling this explodes
        # on large datasets. Hard cap at 2048 each as a safe fallback.
        _n_train_cap = self._subsample_samples or self._subsample_train or 2048
        if len(X_tr_enc) > _n_train_cap:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(X_tr_enc), size=_n_train_cap, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        _n_test_cap = self._subsample_test or 2048
        if len(X_te_enc) > _n_test_cap:
            X_te_enc = X_te_enc[:_n_test_cap]
            y_test    = y_test.iloc[:_n_test_cap]

        X_tr  = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te  = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)
        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        n_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops
        n_cls   = clf.n_classes_
        T       = getattr(clf, "softmax_temperature", 1.0) or 1.0

        # clf (IrisClassifier) is no longer needed — drop it so its GPU tensors
        # (KV caches for n_estimators training subsets) are freed before the
        # first forward pass, which would otherwise OOM on large datasets.
        del clf
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

        def _sync():
            if is_cuda:
                torch.cuda.synchronize()

        def _logits_to_proba(logits: torch.Tensor) -> np.ndarray:
            out = logits[0, :, :n_cls].float().cpu().numpy()
            out = out / max(float(T), 1e-8)
            out -= out.max(axis=-1, keepdims=True)
            exp_out = np.exp(out)
            proba = exp_out / exp_out.sum(axis=-1, keepdims=True)
            return proba / proba.sum(axis=1, keepdims=True)

        def _score(proba: np.ndarray) -> tuple[str, float]:
            y_true = y_test.to_numpy()
            if problem_type == "binary":
                return "roc_auc", float(roc_auc_score(y_true, proba[:, 1]))
            elif problem_type == "multiclass":
                return "log_loss", float(-log_loss(y_true, proba))
            else:
                return "neg_mse", float(-mean_squared_error(y_true, proba[:, 0]))

        def _run_mode(m, input_dtype=None, autocast_dtype=None) -> tuple[str, float, float, float, float, float]:
            """Warm-up once, then n_timing_repeats timed runs.

            Args:
                input_dtype:    cast X_all to this dtype before each forward
                                (INT4 kernels require the input tensor to be BF16).
                autocast_dtype: wrap forward in torch.autocast with this dtype
                                (handles intermediate float32 tensors inside the model
                                that would otherwise mismatch BF16 quantized weights).

            Returns:
                metric_name   : str
                score         : float
                time_ms       : float  — median wall time over timed runs
                peak_mb       : float  — VRAM high-watermark during timed runs
                                        (model weights + activations + shared tensors)
                params_mb     : float  — static parameter storage (sum of p.nbytes())
                activation_mb : float  — peak_mb − params_mb ≈ ephemeral forward overhead
            """
            import contextlib
            X_in = X_all.to(input_dtype).contiguous() if input_dtype is not None else X_all
            _ac = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None and is_cuda
                else contextlib.nullcontext()
            )
            with _ac, torch.no_grad():
                logits = m(X_in, y_full, single_eval_pos=n_train, num_loops=n_loops)
            _sync()

            # Parameter memory: exact byte count from actual storage dtype.
            # For quantized models this reflects INT8/INT4 weight sizes.
            params_mb = sum(p.nbytes for p in m.parameters()) / (1024 ** 2)

            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(n_timing_repeats):
                _sync()
                t0 = time.perf_counter()
                with _ac, torch.no_grad():
                    logits = m(X_in, y_full, single_eval_pos=n_train, num_loops=n_loops)
                _sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            time_ms  = float(np.median(times))
            peak_mb  = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if is_cuda else float("nan")
            )
            activation_mb = peak_mb - params_mb if is_cuda else float("nan")
            metric_name, score = _score(_logits_to_proba(logits))
            return metric_name, score, time_ms, peak_mb, params_mb, activation_mb

        def _ao_import(*names):
            """Resolve torchao quantization symbols across API versions.

            Tries in order:
              1. Direct attribute on torchao.quantization  (function API, older)
              2. torchao.quantization.quant_api            (older sub-module)
              3. Config-class shims                        (newer torchao with
                 Int8WeightOnlyConfig / Int4WeightOnlyConfig / etc.)
            """
            import importlib
            mod = importlib.import_module("torchao.quantization")

            # Config-class shims: wrap new *Config classes to look like old factory fns.
            _shims = {
                "int8_weight_only": lambda: getattr(mod, "Int8WeightOnlyConfig")(),
                "int4_weight_only": lambda group_size=128: getattr(mod, "Int4WeightOnlyConfig")(group_size=group_size),
                "int8_dynamic_activation_int8_weight": lambda: getattr(mod, "Int8DynamicActivationInt8WeightConfig")(),
                "int4_dynamic_activation_int4_weight": lambda: getattr(mod, "Int4WeightOnlyConfig")(),  # best available
            }

            result = []
            for n in names:
                if hasattr(mod, n):
                    result.append(getattr(mod, n))
                    continue
                # Try quant_api sub-module
                try:
                    sub = importlib.import_module("torchao.quantization.quant_api")
                    if hasattr(sub, n):
                        result.append(getattr(sub, n))
                        continue
                except ImportError:
                    pass
                # Fall back to Config shim
                if n in _shims:
                    result.append(_shims[n])
                    continue
                raise ImportError(
                    f"torchao: cannot find '{n}' (tried function API, quant_api, and Config shims)"
                )
            return tuple(result)

        def _quant_filter(mod, fqn: str) -> bool:
            """Skip linear layers too small for INT8/INT4 kernels.

            torchao INT8 requires in_features ≥ 16 (alignment), and INT4
            group-quantization requires in_features % group_size == 0.
            Layers below these thresholds raise shape errors at inference time.
            """
            if isinstance(mod, torch.nn.Linear):
                return mod.in_features >= 16 and mod.out_features >= 16
            return False

        results: dict = {}

        # ── Baseline (bfloat16) ────────────────────────────────────────────
        # Use autocast so activations run in bfloat16, matching predict_proba.
        # Without this the forward runs in float32 (2× larger activations),
        # causing OOM on datasets like Amazon while predict_proba succeeds.
        metric_name, score, time_ms, peak_mb, params_mb, act_mb = _run_mode(
            loopicl, autocast_dtype=torch.bfloat16
        )
        results["metric"]                  = metric_name
        results["baseline_score"]          = score
        results["baseline_time_ms"]        = time_ms
        results["baseline_peak_mb"]        = peak_mb
        results["baseline_params_mb"]      = params_mb
        results["baseline_activation_mb"]  = act_mb
        print(f"        [quant] baseline      {metric_name}={score:.4f}  "
              f"time={time_ms:.1f}ms  "
              f"peak={peak_mb:.0f}MB  params={params_mb:.0f}MB  act={act_mb:.0f}MB")

        # ── Weight-only INT8 ───────────────────────────────────────────────
        # Weights stored as INT8 (2× memory reduction), dequantized to bf16
        # for compute.  Memory-bandwidth win; no extra FLOPs.
        try:
            quantize_, int8_weight_only = _ao_import("quantize_", "int8_weight_only")
            model_w8 = copy.deepcopy(loopicl)
            quantize_(model_w8, int8_weight_only(), filter_fn=_quant_filter)
            _, score_w8, time_w8, peak_w8, params_w8, act_w8 = _run_mode(
                model_w8, autocast_dtype=torch.bfloat16
            )
            del model_w8
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["weight_int8_score"]         = score_w8
            results["weight_int8_time_ms"]       = time_w8
            results["weight_int8_peak_mb"]       = peak_w8
            results["weight_int8_params_mb"]     = params_w8
            results["weight_int8_activation_mb"] = act_w8
            print(f"        [quant] weight_int8   {metric_name}={score_w8:.4f}  "
                  f"time={time_w8:.1f}ms  "
                  f"peak={peak_w8:.0f}MB  params={params_w8:.0f}MB  act={act_w8:.0f}MB  "
                  f"drop={score - score_w8:+.4f}  speedup={time_ms / time_w8:.2f}x")
        except Exception as exc:
            print(f"        [quant] weight_int8   SKIPPED ({exc})")
            results["weight_int8_score"]     = float("nan")
            results["weight_int8_time_ms"]   = float("nan")
            results["weight_int8_memory_mb"] = float("nan")

        # ── Weight-only INT4 ───────────────────────────────────────────────
        # Weights stored as INT4 (~4× memory reduction vs bf16), dequantized
        # to bf16 for compute.  Uses grouped quantization (group_size=128).
        # Note: inner dim must be divisible by group_size; skipped otherwise.
        try:
            # torchao INT4 weight-only kernels use TMA descriptors (CUTLASS 3.x),
            # which require Hopper (SM90+).  A100 is SM80 — skip gracefully.
            if is_cuda:
                _cc = torch.cuda.get_device_capability()  # current device
                if _cc < (9, 0):
                    raise RuntimeError(
                        f"weight_int4 requires Hopper (SM90+) for TMA kernels; "
                        f"got SM{_cc[0]}{_cc[1]} (A100=SM80, H100=SM90)"
                    )
            quantize_, int4_weight_only = _ao_import("quantize_", "int4_weight_only")
            # quantize_() requires BF16 model weights; cast before quantizing.
            # autocast handles intermediate float32 tensors during the forward pass
            # (e.g. from non-matmul ops that stay float32 between quantized layers).
            model_w4 = copy.deepcopy(loopicl).bfloat16()
            quantize_(model_w4, int4_weight_only(group_size=128), filter_fn=_quant_filter)
            _, score_w4, time_w4, peak_w4, params_w4, act_w4 = _run_mode(
                model_w4, input_dtype=torch.bfloat16, autocast_dtype=torch.bfloat16
            )
            del model_w4
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["weight_int4_score"]         = score_w4
            results["weight_int4_time_ms"]       = time_w4
            results["weight_int4_peak_mb"]       = peak_w4
            results["weight_int4_params_mb"]     = params_w4
            results["weight_int4_activation_mb"] = act_w4
            print(f"        [quant] weight_int4   {metric_name}={score_w4:.4f}  "
                  f"time={time_w4:.1f}ms  "
                  f"peak={peak_w4:.0f}MB  params={params_w4:.0f}MB  act={act_w4:.0f}MB  "
                  f"drop={score - score_w4:+.4f}  speedup={time_ms / time_w4:.2f}x")
        except Exception as exc:
            print(f"        [quant] weight_int4   SKIPPED ({exc})")
            results["weight_int4_score"]     = float("nan")
            results["weight_int4_time_ms"]   = float("nan")
            results["weight_int4_memory_mb"] = float("nan")

        # ── Weight-only NF4 via bitsandbytes (A100-compatible INT4) ───────
        # Uses NF4 (4-bit NormalFloat) quantization from bitsandbytes.
        # Works on Ampere (SM80) unlike torchao INT4 which needs Hopper (SM90).
        # Weights stored as NF4, dequantized to bf16 for compute.
        try:
            import bitsandbytes as bnb

            def _replace_with_bnb4(module):
                for name, child in list(module.named_children()):
                    if isinstance(child, torch.nn.Linear) and _quant_filter(child, name):
                        new_layer = bnb.nn.Linear4bit(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            compute_dtype=torch.bfloat16,
                            quant_type="nf4",
                        )
                        new_layer.weight = bnb.nn.Params4bit(
                            child.weight.data.contiguous(),
                            requires_grad=False,
                            quant_type="nf4",
                        )
                        if child.bias is not None:
                            new_layer.bias = child.bias
                        setattr(module, name, new_layer)
                    else:
                        _replace_with_bnb4(child)

            model_nf4 = copy.deepcopy(loopicl).bfloat16()
            _replace_with_bnb4(model_nf4)
            # bitsandbytes quantizes Params4bit on first .cuda() / .to(device) call
            model_nf4.to(device)
            model_nf4.eval()
            _, score_nf4, time_nf4, peak_nf4, params_nf4, act_nf4 = _run_mode(
                model_nf4, input_dtype=torch.bfloat16, autocast_dtype=torch.bfloat16
            )
            del model_nf4
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["weight_nf4_score"]         = score_nf4
            results["weight_nf4_time_ms"]       = time_nf4
            results["weight_nf4_peak_mb"]       = peak_nf4
            results["weight_nf4_params_mb"]     = params_nf4
            results["weight_nf4_activation_mb"] = act_nf4
            print(f"        [quant] weight_nf4    {metric_name}={score_nf4:.4f}  "
                  f"time={time_nf4:.1f}ms  "
                  f"peak={peak_nf4:.0f}MB  params={params_nf4:.0f}MB  act={act_nf4:.0f}MB  "
                  f"drop={score - score_nf4:+.4f}  speedup={time_ms / time_nf4:.2f}x")
        except Exception as exc:
            print(f"        [quant] weight_nf4    SKIPPED ({exc})")
            results["weight_nf4_score"]     = float("nan")
            results["weight_nf4_time_ms"]   = float("nan")
            results["weight_nf4_memory_mb"] = float("nan")

        # ── Compute INT8 (dynamic activation + weight INT8) ────────────────
        # Both weights and activations quantized to INT8 on-the-fly.
        # Uses A100 INT8 Tensor Cores → 2× FP16 throughput.
        try:
            quantize_, int8_dynamic_activation_int8_weight = _ao_import(
                "quantize_", "int8_dynamic_activation_int8_weight"
            )
            model_c8 = copy.deepcopy(loopicl)
            quantize_(model_c8, int8_dynamic_activation_int8_weight(), filter_fn=_quant_filter)
            # torchao quantize_() is designed to be paired with torch.compile;
            # without it the per-token quant overhead dominates.
            model_c8 = torch.compile(model_c8)
            _, score_c8, time_c8, peak_c8, params_c8, act_c8 = _run_mode(
                model_c8, autocast_dtype=torch.bfloat16
            )
            del model_c8
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["compute_int8_score"]         = score_c8
            results["compute_int8_time_ms"]       = time_c8
            results["compute_int8_peak_mb"]       = peak_c8
            results["compute_int8_params_mb"]     = params_c8
            results["compute_int8_activation_mb"] = act_c8
            print(f"        [quant] compute_int8  {metric_name}={score_c8:.4f}  "
                  f"time={time_c8:.1f}ms  "
                  f"peak={peak_c8:.0f}MB  params={params_c8:.0f}MB  act={act_c8:.0f}MB  "
                  f"drop={score - score_c8:+.4f}  speedup={time_ms / time_c8:.2f}x")
        except Exception as exc:
            print(f"        [quant] compute_int8  SKIPPED ({exc})")
            results["compute_int8_score"]     = float("nan")
            results["compute_int8_time_ms"]   = float("nan")
            results["compute_int8_memory_mb"] = float("nan")

        # ── Compute INT4 (INT8 activations + INT4 weights) ─────────────────
        # Uses Int8DynActInt4WeightQuantizer: activations dynamically quantized
        # to INT8, weights to INT4.  On A100 this uses INT8 Tensor Cores.
        # (True INT4×INT4 compute requires H100 / FP8 path.)
        # Falls back to quantize_() with Int8DynActInt4WeightConfig if available,
        # since the Quantizer class API does not support filter_fn.
        try:
            if is_cuda:
                _cc = torch.cuda.get_device_capability()  # current device
                if _cc < (9, 0):
                    raise RuntimeError(
                        f"compute_int4 requires Hopper (SM90+) for TMA kernels; "
                        f"got SM{_cc[0]}{_cc[1]} (A100=SM80, H100=SM90)"
                    )
            import importlib
            _ao_mod = importlib.import_module("torchao.quantization")
            model_c4 = copy.deepcopy(loopicl)
            if hasattr(_ao_mod, "Int8DynActInt4WeightConfig"):
                # Newer torchao: use quantize_() with filter_fn support
                quantize_, _cfg = _ao_import("quantize_", "int8_dynamic_activation_int8_weight")
                quantize_(model_c4, _cfg(), filter_fn=_quant_filter)
            else:
                # Older torchao: use Quantizer class (no filter_fn).
                # Pre-check: all linear layers must have in_features divisible by groupsize.
                _Quantizer = getattr(_ao_mod, "Int8DynActInt4WeightQuantizer")
                _groupsize = 256  # default groupsize in Int8DynActInt4WeightQuantizer
                _bad = [
                    (n, m.in_features) for n, m in model_c4.named_modules()
                    if isinstance(m, torch.nn.Linear) and m.in_features % _groupsize != 0
                ]
                if _bad:
                    raise RuntimeError(
                        f"{len(_bad)} layers have in_features not divisible by "
                        f"groupsize={_groupsize} (e.g. '{_bad[0][0]}' in_features={_bad[0][1]}). "
                        f"Upgrade torchao for Int8DynActInt4WeightConfig with filter_fn support."
                    )
                model_c4 = _Quantizer().quantize(model_c4)
            model_c4.eval()
            _, score_c4, time_c4, peak_c4, params_c4, act_c4 = _run_mode(
                model_c4, input_dtype=torch.bfloat16, autocast_dtype=torch.bfloat16
            )
            del model_c4
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["compute_int4_score"]         = score_c4
            results["compute_int4_time_ms"]       = time_c4
            results["compute_int4_peak_mb"]       = peak_c4
            results["compute_int4_params_mb"]     = params_c4
            results["compute_int4_activation_mb"] = act_c4
            print(f"        [quant] compute_int4  {metric_name}={score_c4:.4f}  "
                  f"time={time_c4:.1f}ms  "
                  f"peak={peak_c4:.0f}MB  params={params_c4:.0f}MB  act={act_c4:.0f}MB  "
                  f"drop={score - score_c4:+.4f}  speedup={time_ms / time_c4:.2f}x")
        except Exception as exc:
            print(f"        [quant] compute_int4  SKIPPED ({exc})")
            results["compute_int4_score"]     = float("nan")
            results["compute_int4_time_ms"]   = float("nan")
            results["compute_int4_memory_mb"] = float("nan")

        # ── Compute INT8 Static (calibrated activation scales) ──────────────
        # Calibration-based static INT8 without PT2E:
        #   1. Forward hook collects per-layer activation absmax in one pass.
        #   2. Each qualifying nn.Linear is replaced with _StaticInt8Linear,
        #      which uses the frozen scale (no runtime amax) and
        #      torch._int_mm for A100 SM80 INT8 Tensor Core matmuls (INT32 acc).
        # Eliminates the per-token kernel-launch overhead of dynamic INT8.
        try:
            class _StaticInt8Linear(torch.nn.Module):
                """nn.Linear replacement: pre-quantized INT8 weights +
                calibrated activation scale → torch._int_mm.

                torch._int_mm requires K and N to be multiples of 8, and M > 16.
                Weights are pre-transposed and padded at construction so the
                hot path only pads activations (cheap) and slices output back.
                Small-M inputs (M ≤ 16) are row-padded to 17 to stay on the
                INT8 path rather than falling back to float dequant.
                """

                def __init__(self, w_int8, bias, scale_w: float, scale_a: float):
                    super().__init__()
                    n_out, k = w_int8.shape
                    # Pad K and N up to next multiple of 16 for torch._int_mm.
                    # cuBLASLt INT8 kernels require 16-element alignment for K and N
                    # (multiples of 8 are insufficient on non-A100 GPUs).
                    # F.pad does not support INT8; use new_zeros + copy instead.
                    k_pad = (-k) % 16
                    n_pad = (-n_out) % 16
                    if k_pad > 0 or n_pad > 0:
                        w_padded = w_int8.new_zeros(n_out + n_pad, k + k_pad)
                        w_padded[:n_out, :k] = w_int8
                        w_int8 = w_padded  # [n_out+n_pad, k+k_pad] int8
                    # Store pre-transposed contiguous weights [K_pad, N_pad] so
                    # forward never creates a non-contiguous view.
                    self.register_buffer("w_int8_t", w_int8.t().contiguous())  # [K_pad, N_pad]
                    self.register_buffer("bias_buf", bias)   # [out] bf16 | None
                    self.scale_w = scale_w
                    self.scale_a = scale_a
                    self.k_pad  = k_pad
                    self.n_out  = n_out   # original out_features for slicing

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    orig = x.shape
                    x_2d = x.view(-1, orig[-1])               # [M, K]
                    M = x_2d.shape[0]

                    # torch._int_mm has two hard requirements on M:
                    #   1. PyTorch check: M > 16  (minimum valid value = 17)
                    #   2. cuBLAS INT8 alignment: M must be a multiple of 16
                    # Combined minimum is 32 (next multiple of 16 that is > 16).
                    M_aligned = max(32, ((M + 15) // 16) * 16)
                    m_pad = M_aligned - M
                    if m_pad:
                        x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, m_pad))

                    x_int8 = (
                        x_2d.float().div_(self.scale_a)
                        .clamp_(-128, 127).to(torch.int8).contiguous()
                    )
                    if self.k_pad > 0:
                        x_padded = x_int8.new_zeros(x_int8.shape[0], x_int8.shape[1] + self.k_pad)
                        x_padded[:, :x_int8.shape[1]] = x_int8
                        x_int8 = x_padded
                    # INT8 × INT8 → INT32 (A100 SM80 Tensor Cores).
                    out = torch._int_mm(x_int8, self.w_int8_t)    # [M_aligned, N_pad] int32
                    out = out.float().mul_(self.scale_a * self.scale_w).to(x.dtype)
                    out = out[:M, :self.n_out]                     # slice M and N padding
                    out = out.view(*orig[:-1], self.n_out)
                    if self.bias_buf is not None:
                        out = out + self.bias_buf.to(out.dtype)
                    return out

            # Step 1 — calibration: collect per-layer activation absmax.
            _act_scales: dict[str, float] = {}
            _hooks = []
            for _nm, _md in loopicl.named_modules():
                if isinstance(_md, torch.nn.Linear) and _quant_filter(_md, _nm):
                    def _make_hook(name):
                        def _h(m, inp, _out):
                            _act_scales[name] = (
                                inp[0].detach().float().abs().amax().item()
                            )
                        return _h
                    _hooks.append(_md.register_forward_hook(_make_hook(_nm)))
            _ac_cal = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if is_cuda else contextlib.nullcontext()
            )
            with _ac_cal, torch.no_grad():
                loopicl(X_all, y_full, single_eval_pos=n_train, num_loops=n_loops)
            for _h in _hooks:
                _h.remove()

            # Step 2 — replace qualifying nn.Linear with _StaticInt8Linear.
            model_c8s = copy.deepcopy(loopicl)
            for _nm, _md in list(model_c8s.named_modules()):
                if (not isinstance(_md, torch.nn.Linear)
                        or not _quant_filter(_md, _nm)
                        or _act_scales.get(_nm, 0.0) == 0.0):
                    continue
                _scale_a = _act_scales[_nm] / 127.0
                _w = _md.weight.detach().float()
                _scale_w = _w.abs().amax().item() / 127.0
                if _scale_w == 0.0:
                    continue
                _w_int8 = _w.div_(_scale_w).clamp_(-128, 127).to(torch.int8)
                _bias = (
                    _md.bias.detach().to(device=device, dtype=torch.bfloat16)
                    if _md.bias is not None else None
                )
                _sl = _StaticInt8Linear(_w_int8, _bias, _scale_w, _scale_a)
                _parent = model_c8s
                for _p in _nm.split(".")[:-1]:
                    _parent = getattr(_parent, _p)
                setattr(_parent, _nm.split(".")[-1], _sl)

            # Step 3 — time the static INT8 model.
            # torch.compile fuses the activation cast chain (bf16→f32→i8→i32→f32→bf16)
            # into fewer kernels and eliminates Python dispatch overhead per layer.
            # The warmup run in _run_mode absorbs compile time.
            # _run_mode counts only nn.Parameters; INT8 weights live in buffers,
            # so recompute params_mb to include both after timing.
            model_c8s = torch.compile(model_c8s)
            _, score_c8s, time_c8s, peak_c8s, _p, _a = _run_mode(
                model_c8s, autocast_dtype=torch.bfloat16
            )
            params_c8s = sum(
                t.nbytes for t in [*model_c8s.parameters(), *model_c8s.buffers()]
            ) / (1024 ** 2)
            act_c8s = peak_c8s - params_c8s

            del model_c8s
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["compute_int8_static_score"]         = score_c8s
            results["compute_int8_static_time_ms"]       = time_c8s
            results["compute_int8_static_peak_mb"]       = peak_c8s
            results["compute_int8_static_params_mb"]     = params_c8s
            results["compute_int8_static_activation_mb"] = act_c8s
            print(
                f"        [quant] compute_int8_static  {metric_name}={score_c8s:.4f}  "
                f"time={time_c8s:.1f}ms  "
                f"peak={peak_c8s:.0f}MB  params={params_c8s:.0f}MB  act={act_c8s:.0f}MB  "
                f"drop={score - score_c8s:+.4f}  speedup={time_ms / time_c8s:.2f}x"
            )
        except Exception as exc:
            print(f"        [quant] compute_int8_static  SKIPPED ({exc})")
            results["compute_int8_static_score"]         = float("nan")
            results["compute_int8_static_time_ms"]       = float("nan")
            results["compute_int8_static_peak_mb"]       = float("nan")
            results["compute_int8_static_params_mb"]     = float("nan")
            results["compute_int8_static_activation_mb"] = float("nan")

        # ── MLP blocks compiled with TensorRT ──────────────────────────────
        # Compiles only MLP / GatedMLP submodules via torch_tensorrt, leaving
        # attention and other layers in eager BF16.  This sidesteps TRT
        # graph-break issues from dynamic masking in attention while still
        # covering the most FLOP-intensive layers (the two large GEMMs per
        # transformer block).  The parent model runs eagerly; compiled blocks
        # dispatch to TRT for their portion only.
        try:
            import torch_tensorrt  # noqa: F401 – import check; used via backend=
            model_trt = copy.deepcopy(loopicl)
            _mlp_names = ("MLP", "GatedMLP")
            _n_compiled = 0
            for _nm, _md in list(model_trt.named_modules()):
                if type(_md).__name__ not in _mlp_names:
                    continue
                _compiled_md = torch.compile(
                    _md,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": {torch.float16},
                        "truncate_long_and_double": True,
                    },
                )
                _parent = model_trt
                for _p in _nm.split(".")[:-1]:
                    _parent = getattr(_parent, _p)
                setattr(_parent, _nm.split(".")[-1], _compiled_md)
                _n_compiled += 1
            print(f"        [quant] mlp_trt  compiling {_n_compiled} MLP/GatedMLP blocks …")
            _, score_trt, time_trt, peak_trt, params_trt, act_trt = _run_mode(
                model_trt, autocast_dtype=torch.bfloat16
            )
            del model_trt
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
            results["mlp_trt_score"]         = score_trt
            results["mlp_trt_time_ms"]       = time_trt
            results["mlp_trt_peak_mb"]       = peak_trt
            results["mlp_trt_params_mb"]     = params_trt
            results["mlp_trt_activation_mb"] = act_trt
            print(
                f"        [quant] mlp_trt  {metric_name}={score_trt:.4f}  "
                f"time={time_trt:.1f}ms  "
                f"peak={peak_trt:.0f}MB  params={params_trt:.0f}MB  act={act_trt:.0f}MB  "
                f"drop={score - score_trt:+.4f}  speedup={time_ms / time_trt:.2f}x"
            )
        except Exception as exc:
            print(f"        [quant] mlp_trt  SKIPPED ({exc})")
            results["mlp_trt_score"]         = float("nan")
            results["mlp_trt_time_ms"]       = float("nan")
            results["mlp_trt_peak_mb"]       = float("nan")
            results["mlp_trt_params_mb"]     = float("nan")
            results["mlp_trt_activation_mb"] = float("nan")

        self.quantization_metrics = results

    def compute_attention_head_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Capture per-head attention statistics and/or head ablation importance.

        Controlled by the flags set at construction time:
          capture_attention_heads  → runs one forward pass with hook captures
          capture_head_ablation    → runs n_blocks × n_heads + 1 forward passes

        Results stored in ``self.attention_head_metrics`` as a dict with keys:
          "stats"    : output of forward_with_attention_head_stats (or None)
          "ablation" : output of forward_with_head_ablation       (or None)

        See paper/notebooks/attention_heads/ for analysis of the resulting data.
        """
        import torch
        from models.loopicl_attention_heads_forward import (
            forward_with_attention_head_stats,
            forward_with_head_ablation,
        )

        clf     = self._model
        loopicl = self._loopicl
        device  = self._device

        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)

        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)
        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        X_tr   = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te   = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all  = torch.cat([X_tr, X_te], dim=1)
        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        model_dtype = next(loopicl.parameters()).dtype
        X_all  = X_all.to(model_dtype)

        num_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops

        stats_result    = None
        ablation_result = None

        if self.capture_attention_heads:
            stats_result = forward_with_attention_head_stats(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
            )

        if self.capture_head_ablation:
            y_te_int = clf.y_encoder_.transform(y_test).astype(np.int64)
            ablation_result = forward_with_head_ablation(
                loopicl, X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
                y_te_int=y_te_int,
                n_classes=clf.n_classes_,
            )

        self.attention_head_metrics = {
            "stats":    stats_result,
            "ablation": ablation_result,
        }


# ---------------------------------------------------------------------------
# Ablation-specific subclasses (each loads its own LoopICL variant)
# ---------------------------------------------------------------------------


def _load_ablation_model(cls_import_path: tuple[str, str], weights_path: str, device: str):
    """Generic loader for ablation model classes."""
    import importlib
    module = importlib.import_module(cls_import_path[0])
    cls = getattr(module, cls_import_path[1])
    checkpoint = torch.load(weights_path, map_location="cpu")
    cfg = dict(checkpoint["config"])
    if "max_classes" in cfg and "max_num_classes" not in cfg:
        cfg["max_num_classes"] = cfg.pop("max_classes")
    model = cls(**cfg)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    del checkpoint
    model.to(device).eval()
    return model


class LoopICLNormHistModel(LoopICLModel):
    name = "LoopICLNormHist"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_norm_hist", "LoopICLNormHist"),
            weights_path, device,
        )


class LoopICLNormHistNoShiftModel(LoopICLModel):
    name = "LoopICLNormHistNoShift"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_norm_hist_no_shift", "LoopICLNormHistNoShift"),
            weights_path, device,
        )


class LoopICLNormHistChunkModel(LoopICLModel):
    name = "LoopICLNormHistChunk"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_norm_hist_chunk", "LoopICLNormHistChunk"),
            weights_path, device,
        )


class LoopICLSandwichModel(LoopICLModel):
    name = "LoopICLSandwich"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_sandwich", "LoopICLSandwich"),
            weights_path, device,
        )


class PairedColLoopICLModel(LoopICLModel):
    name = "PairedColLoopICL"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_paired_col", "PairedColLoopICL"),
            weights_path, device,
        )


class LabelPairedLoopICLModel(LoopICLModel):
    name = "LabelPairedLoopICL"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_paired_col", "LabelPairedLoopICL"),
            weights_path, device,
        )


class LoopICLUnrolledModel(LoopICLModel):
    name = "LoopICLUnrolled"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model", "LoopICLUnrolled"),
            weights_path, device,
        )


class QuadColLoopICLModel(LoopICLModel):
    name = "QuadColLoopICL"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_quad_col", "QuadColLoopICL"),
            weights_path, device,
        )


class TabFMEmbedLoopICLModel(LoopICLModel):
    name = "TabFMEmbedLoopICL"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_tabfm_embed", "TabFMEmbedLoopICL"),
            weights_path, device,
        )


class FeatCircShiftLoopICLModel(LoopICLModel):
    name = "FeatCircShiftLoopICL"
    def _load_weights(self, weights_path, device):
        return _load_ablation_model(
            ("loopicl.models.loopicl.model_feat_circ", "FeatCircShiftLoopICL"),
            weights_path, device,
        )


# ---------------------------------------------------------------------------
# LoopICLEarlyExit — run all loops and capture gate λ / CDF per loop
# ---------------------------------------------------------------------------


class LoopICLEarlyExitModel(LoopICLModel):
    """LoopICL early-exit model that captures gate λ values per loop.

    Accepts all standard LoopICLModel fields plus:

    exit_q : float
        Quantile threshold used during normal predict_proba inference
        (0 = exit as early as possible, 1 = run all loops). Default 1.0.
    capture_exit_steps : bool
        If True, ``compute_exit_gate_values(X_train, y_train, X_test)``
        becomes available after ``fit()``.  Runs all loops with
        ``return_exit_info=True`` so the gate's λ and CDF values are
        captured without applying any exit threshold, enabling offline
        simulation of any exit_q.
    """

    name = "LoopICLEarlyExit"

    def __init__(
        self,
        exit_q: float = 1.0,
        capture_exit_steps: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._exit_q = exit_q
        self.capture_exit_steps = capture_exit_steps
        self.exit_gate_values: dict | None = None

    def _load_weights(self, weights_path: str, device: str) -> torch.nn.Module:
        model = _load_ablation_model(
            ("loopicl.models.loopicl.model_early_exit", "LoopICLEarlyExit"),
            weights_path, device,
        )
        # Apply config-specified exit_q (overrides the checkpoint's stored default)
        model.exit_q = self._exit_q
        return model

    def compute_exit_gate_values(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> None:
        """Single B=1 forward with return_exit_info=True.

        Captures the gate's per-loop λ and cumulative CDF so exit thresholds
        can be applied offline for any exit_q.

        Results
        -------
        self.exit_gate_values : dict
            "lambda" : np.ndarray (n_loops,) — λₜ per loop (gate exit prob.)
            "cdf"    : np.ndarray (n_loops,) — CDF_t = 1 − Π_{j≤t}(1 − λⱼ)
        self.loop_logits : list[np.ndarray]
            Per-loop prediction probabilities for test rows,
            each shape (n_test, n_classes).  Index 0 = first loop.
        """
        clf     = self._model
        device  = self._device
        loopicl = self._loopicl

        X_train = X_train.copy(); X_train.columns = X_train.columns.astype(str)
        X_test  = X_test.copy();  X_test.columns  = X_test.columns.astype(str)
        X_tr_enc = clf.X_encoder_.transform(X_train)
        X_te_enc = clf.X_encoder_.transform(X_test)
        y_tr_enc = clf.y_encoder_.transform(y_train).astype(np.float32)

        n_train_full = len(X_tr_enc)
        if self._subsample_samples is not None and self._subsample_samples < n_train_full:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_train_full, size=self._subsample_samples, replace=False))
            X_tr_enc = X_tr_enc[idx]
            y_tr_enc = y_tr_enc[idx]

        X_tr  = torch.tensor(X_tr_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_te  = torch.tensor(X_te_enc, dtype=torch.float32, device=device).unsqueeze(0)
        X_all = torch.cat([X_tr, X_te], dim=1)

        n_train = X_tr.shape[1]
        n_total = X_all.shape[1]

        y_tr   = torch.tensor(y_tr_enc, dtype=torch.long, device=device).unsqueeze(0)
        y_full = torch.zeros(1, n_total, dtype=torch.long, device=device)
        y_full[:, :n_train] = y_tr

        num_loops = self._num_loops if self._num_loops is not None else loopicl.num_icl_loops

        with torch.no_grad():
            all_logits, all_lambdas = loopicl(
                X_all, y_full,
                single_eval_pos=n_train,
                num_loops=num_loops,
                return_exit_info=True,   # disables early exit, returns all λ values
            )

        # all_lambdas: (n_loops, B=1) → (n_loops,)
        lambdas_np = all_lambdas[:, 0].float().cpu().numpy()

        # CDF_t = 1 − Π_{j≤t}(1 − λⱼ)
        survival = np.cumprod(1.0 - lambdas_np)
        cdf_np   = 1.0 - survival

        T     = clf.softmax_temperature
        n_cls = clf.n_classes_

        def _to_proba(logits_tensor) -> np.ndarray:
            out = logits_tensor[0, :, :n_cls].float().cpu().numpy()
            out = out / max(T, 1e-8)
            out = out - out.max(axis=-1, keepdims=True)
            exp_out = np.exp(out)
            proba = exp_out / exp_out.sum(axis=-1, keepdims=True)
            return proba / proba.sum(axis=1, keepdims=True)

        self.exit_gate_values = {"lambda": lambdas_np, "cdf": cdf_np}
        self.loop_logits      = [_to_proba(lg) for lg in all_logits]
        self.loop_embeddings  = None
