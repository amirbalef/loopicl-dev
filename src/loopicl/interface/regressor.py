"""Sklearn-compatible regressor interface for the LoopICL tabular foundation model.

Uses a quantile / pinball decoder:
  - Model: ``LoopICLTabICLReg`` — outputs ``(B, n_test, num_quantiles)`` raw quantile values
  - Ensemble: inverse-transform per-estimator quantiles to z-norm space → average
  - Decode: ``QuantileDistribution.{mean,variance,...}`` with tail extrapolation

Output types
------------
``"mean"``      – E[Y]
``"median"``    – 0.5-quantile
``"std"``       – standard deviation
``"quantiles"`` – list of arrays, one per quantile in ``quantiles``
``"main"``      – dict with mean, median, quantiles
``"full"``      – main + raw distribution object

Usage::

    from loopicl.interface.regressor import LoopICLRegressor

    reg = LoopICLRegressor(model=model, device="cuda", n_estimators=8)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)                    # mean by default
    main  = reg.predict(X_test, output_type="main")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional, Union

import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

from loopicl.interface._preprocessing import (
    EnsembleGenerator,
    XEncoder,
    _has_prefill,
    _quantize_cache,
    _wrapper_prefill,
    _wrapper_predict_cached,
)
_NORM_METHODS = ["squashing_scaler_default", "quantile_uni_append"]

_Y_TRANSFORM_METHODS = [None, "safepower"]


# ---------------------------------------------------------------------------
# LoopICLRegressor
# ---------------------------------------------------------------------------

OutputType = Literal["mean", "median", "std", "quantiles", "main", "full"]


class LoopICLRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible regressor wrapping ``LoopICLTabICLReg`` (quantile / pinball decoder).

    Works with models that output raw quantile predictions directly
    (``num_quantiles`` evenly-spaced quantile values, trained with pinball loss).

    Ensemble strategy
    -----------------
    - Same X-normalisation / feature-shuffle / y-transform ensemble
    - Per estimator: model outputs ``(B, n_test, num_quantiles)`` quantile values in
      the (optionally power-transformed) z-norm y-space
    - For power-transformed estimators: inverse-transform predictions back to z-norm space
    - Average raw quantile values across estimators in z-norm space
    - Scale from z-norm → original y-space (``q * y_std_ + y_mean_``)
    - Wrap in ``QuantileDistribution`` for analytical mean, variance, quantile decode

    Parameters
    ----------
    model : LoopICLTabICLReg
        The regression model.
    n_estimators : int, default=8
    norm_methods : list[str] or None
    y_transform_methods : list or None
    feat_shuffle_method : str, default='balanced_shuffle'
    outlier_threshold : float or None, default=None
    fingerprint_feature : bool, default=True
    batch_size : int or None, default=8
    test_chunk_size : int or None, default=None
        Maximum test samples per forward call. The test set is split into
        chunks of this size and processed sequentially, with each chunk
        concatenated to the full train context. Reduces peak VRAM when
        n_test is large (memory scales quadratically with sequence length).
        ``None`` processes all test samples in one call.
    cache_context : bool, default=False
        When True and the model supports ``prefill`` / ``predict_cached``, the
        train-side KV cache is built once at ``fit()`` time and reused across all
        ``predict()`` calls.  Per-estimator y-transforms are also pre-computed.
        Automatically disables the fingerprint feature.
    quantize_kv_cache : bool, default=False
        When True, int8-quantizes the KV cache after prefill via
        ``cache.quantize()`` if supported.  Reduces peak VRAM ~4×.
    keep_cache_on_cpu : bool, default=False
        Move the KV cache to CPU immediately after prefill and restore it to
        GPU before each decode call.
    subsample_samples : int or None, default=None
    max_features : int or None, default=None
    random_state : int or None, default=42
    tail_type : str, default='exp'
        Tail extrapolation used by ``QuantileDistribution``: ``'exp'`` or ``'gpd'``.
    """

    def __init__(
        self,
        model,
        device=None,
        dtype: str = "float32",
        num_loops: Optional[int] = None,
        n_estimators: int = 8,
        norm_methods: Optional[list] = None,
        y_transform_methods: Optional[list] = None,
        feat_shuffle_method: str = "balanced_shuffle",
        outlier_threshold: Optional[float] = None,
        fingerprint_feature: bool = True,
        batch_size: Optional[int] = 8,
        test_chunk_size: Optional[int] = None,
        cache_context: bool = False,
        quantize_kv_cache: bool = False,
        keep_cache_on_cpu: bool = False,
        subsample_samples: Optional[int] = None,
        max_features: Optional[int] = None,
        random_state: Optional[int] = 42,
        tail_type: str = "exp",
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.num_loops = num_loops
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.y_transform_methods = y_transform_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.fingerprint_feature = fingerprint_feature
        self.batch_size = batch_size
        self.test_chunk_size = test_chunk_size
        self.cache_context = cache_context
        self.quantize_kv_cache = quantize_kv_cache
        self.keep_cache_on_cpu = keep_cache_on_cpu
        self.subsample_samples = subsample_samples
        self.max_features = max_features
        self.random_state = random_state
        self.tail_type = tail_type

    def _resolve_device(self):
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _call_model(self, X_np: np.ndarray, y_np: np.ndarray, eval_pos: int) -> np.ndarray:
        """Convert numpy arrays to tensors, run the model, return (B, M, out) numpy array."""
        import torch as _torch
        device = self._resolve_device()
        X_t = _torch.as_tensor(X_np.astype(np.float32), device=device)
        y_t = _torch.as_tensor(y_np.astype(np.float32), device=device)

        _dtype_map = {"float16": _torch.float16, "bfloat16": _torch.bfloat16}
        amp_dtype = _dtype_map.get(self.dtype)

        kwargs: dict = {}
        if self.num_loops is not None:
            kwargs["num_loops"] = self.num_loops

        batched = X_t.ndim == 3
        if not batched:
            X_t = X_t.unsqueeze(0)
            y_t = y_t.unsqueeze(0)

        with _torch.no_grad():
            if amp_dtype is not None and "cuda" in str(device):
                with _torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(X_t, y_t, **kwargs)
            else:
                out = self.model(X_t, y_t, **kwargs)

        if not batched:
            out = out.squeeze(0)
        return out.float().cpu().numpy()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoopICLRegressor":
        self.X_encoder_ = XEncoder()
        X_enc = self.X_encoder_.fit_transform(X)

        y_enc = np.asarray(y, dtype=np.float32).ravel()
        self.y_mean_ = float(np.mean(y_enc))
        self.y_std_ = max(float(np.std(y_enc)), 1e-9)
        y_scaled = (y_enc - self.y_mean_) / self.y_std_

        self.n_features_in_ = X_enc.shape[1]
        self.n_train_samples_ = X_enc.shape[0]
        self.n_outputs_ = 1

        norm_methods = self.norm_methods if self.norm_methods is not None else _NORM_METHODS

        # cache_context requires fingerprint=False.
        fingerprint_feature = self.fingerprint_feature and not self.cache_context

        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method="none",
            outlier_threshold=self.outlier_threshold,
            subsample_samples=self.subsample_samples,
            max_features=self.max_features,
            fingerprint_feature=fingerprint_feature,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X_enc, y_scaled)

        if self.cache_context and _has_prefill(self.model):
            self.context_caches_, self._context_cache_batch_size_ = (
                self._build_context_cache()
            )

        return self

    # ------------------------------------------------------------------
    # _build_context_cache
    # ------------------------------------------------------------------

    def _build_context_cache(self) -> tuple:
        """Prefill the train-side KV cache and pre-compute y-transforms.

        Returns
        -------
        caches     : list of KV cache objects in shape-group/batch order
        batch_size : int
        """
        all_X_train, all_y_cont = self.ensemble_generator_.transform_train_only()
        n_estimators = len(all_X_train)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        y_transform_methods = (
            self.y_transform_methods if self.y_transform_methods is not None
            else _Y_TRANSFORM_METHODS
        )

        all_y_model: list = []
        all_power_tfms: list = []
        for est_idx, y_cont in enumerate(all_y_cont):
            ytm = y_transform_methods[est_idx % len(y_transform_methods)]
            if ytm is None:
                all_y_model.append(y_cont)
                all_power_tfms.append(None)
            elif ytm == "safepower":
                power_tfm = PowerTransformer(method="yeo-johnson", standardize=True)
                y_model = power_tfm.fit_transform(
                    y_cont.reshape(-1, 1)
                ).ravel().astype(np.float32)
                all_y_model.append(y_model)
                all_power_tfms.append(power_tfm)
            else:
                raise ValueError(f"Unknown y_transform_method {ytm!r}.")

        # Store pre-computed transforms for predict() reuse.
        self._cache_power_tfms_ = all_power_tfms

        shape_groups: dict = defaultdict(list)
        for i, X_i in enumerate(all_X_train):
            shape_groups[X_i.shape[1]].append(i)

        caches: list = []
        for _n_feats, indices in shape_groups.items():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start: start + batch_size]
                batch_X_train = np.stack(
                    [all_X_train[i] for i in batch_idx], axis=0
                )
                batch_y = np.stack([all_y_model[i] for i in batch_idx], axis=0)

                cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
                if self.quantize_kv_cache:
                    cache = _quantize_cache(cache)
                if self.keep_cache_on_cpu:
                    cache = cache.to("cpu")
                caches.append(cache)

        return caches, batch_size

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        output_type: OutputType = "mean",
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> Union[np.ndarray, list, dict]:
        """Predict using the TabICL quantile decoder.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)
        output_type : str, default='mean'
            One of ``'mean'``, ``'median'``, ``'std'``, ``'quantiles'``,
            ``'main'``, ``'full'``.
        quantiles : tuple of float, default=(0.1, 0.5, 0.9)
            Quantile levels for ``output_type='quantiles'`` / ``'main'`` / ``'full'``.

        Returns
        -------
        np.ndarray  for scalar output types (mean, median, std)
        list        for 'quantiles'
        dict        for 'main' / 'full'
        """
        from loopicl.models.quantile_dist import QuantileDistribution

        check_is_fitted(self, ["ensemble_generator_", "y_mean_", "y_std_"])

        X_enc = self.X_encoder_.transform(X)
        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_

        ensemble_data = self.ensemble_generator_.transform(X_enc)

        all_X: list[np.ndarray] = []
        all_y_cont: list[np.ndarray] = []

        for _norm_method, (X_ens, y_ens) in ensemble_data.items():
            for i in range(X_ens.shape[0]):
                all_X.append(X_ens[i])
                all_y_cont.append(y_ens[i].astype(np.float32))

        n_estimators = len(all_X)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        y_transform_methods = (
            self.y_transform_methods if self.y_transform_methods is not None
            else _Y_TRANSFORM_METHODS
        )

        use_cache_context = self.cache_context and hasattr(self, "context_caches_")

        if use_cache_context:
            # Reuse pre-computed transforms from fit() time.
            all_y_model: list = list(all_y_cont)
            all_power_tfms: list = self._cache_power_tfms_
        else:
            all_y_model = []
            all_power_tfms = []
            for est_idx, y_cont in enumerate(all_y_cont):
                ytm = y_transform_methods[est_idx % len(y_transform_methods)]
                if ytm is None:
                    all_y_model.append(y_cont)
                    all_power_tfms.append(None)
                elif ytm == "safepower":
                    power_tfm = PowerTransformer(method="yeo-johnson", standardize=True)
                    y_model = power_tfm.fit_transform(
                        y_cont.reshape(-1, 1)
                    ).ravel().astype(np.float32)
                    all_y_model.append(y_model)
                    all_power_tfms.append(power_tfm)
                else:
                    raise ValueError(f"Unknown y_transform_method {ytm!r}.")

        # Accumulate raw quantile values in z-norm space.
        accumulated_quantiles: np.ndarray | None = None
        num_quantiles: int | None = None
        test_chunk_size = self.test_chunk_size if self.test_chunk_size is not None else n_test

        use_two_phase = _has_prefill(self.model) and (
            self.test_chunk_size is not None or use_cache_context
        )

        shape_groups: dict = defaultdict(list)
        for i, X_i in enumerate(all_X):
            shape_groups[X_i.shape[1]].append(i)

        _cache_idx = 0
        _cache_batch_size = (
            self._context_cache_batch_size_
            if use_cache_context
            else batch_size
        )

        for _n_feats, indices in shape_groups.items():
            for start in range(0, len(indices), _cache_batch_size):
                batch_idx = indices[start: start + _cache_batch_size]

                batch_X_train = np.stack(
                    [all_X[i][:n_train] for i in batch_idx], axis=0
                )
                batch_y = np.stack([all_y_model[i] for i in batch_idx], axis=0)

                if use_cache_context:
                    cache = self.context_caches_[_cache_idx]
                    _cache_idx += 1
                elif use_two_phase:
                    cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
                    if self.quantize_kv_cache:
                        cache = _quantize_cache(cache)
                    if self.keep_cache_on_cpu:
                        cache = cache.to("cpu")

                for t_start in range(0, n_test, test_chunk_size):
                    t_end = min(t_start + test_chunk_size, n_test)
                    chunk = t_end - t_start

                    batch_X_chunk = np.stack(
                        [all_X[i][n_train + t_start: n_train + t_end] for i in batch_idx],
                        axis=0,
                    )

                    if use_two_phase:
                        if self.keep_cache_on_cpu:
                            device = self._resolve_device()
                            cache = cache.to(device)
                        raw = _wrapper_predict_cached(
                            self.model, self._resolve_device(), self.dtype, batch_X_chunk, cache
                        )
                        if self.keep_cache_on_cpu:
                            cache = cache.to("cpu")
                    else:
                        batch_X = np.concatenate([batch_X_train, batch_X_chunk], axis=1)
                        raw = self._call_model(batch_X, batch_y, eval_pos=n_train)
                    # raw: (B, chunk, num_quantiles)

                    if accumulated_quantiles is None:
                        num_quantiles = raw.shape[-1]
                        accumulated_quantiles = np.zeros(
                            (n_test, num_quantiles), dtype=np.float64
                        )

                    for j, est_idx in enumerate(batch_idx):
                        q_vals = raw[j]  # (chunk, num_quantiles) in (power-)transformed z-norm
                        ptfm = all_power_tfms[est_idx]
                        if ptfm is not None:
                            # Yeo-Johnson inverse is only numerically stable when λ ∈ [0, 2].
                            # Outside this range the formula raises a negative base to a
                            # non-integer power → NaN.  Fall back to raw z-norm predictions
                            # for those estimators rather than filling with ±1e15.
                            lambda_safe = all(0.0 <= lam <= 2.0 for lam in ptfm.lambdas_)
                            if not lambda_safe:
                                accumulated_quantiles[t_start:t_end] += q_vals.astype(np.float64)
                            else:
                                # Inverse power-transform: move from power-y-space → z-norm space.
                                q_clamp = np.clip(q_vals, -1e6, 1e6).astype(np.float64)
                                q_znorm = ptfm.inverse_transform(
                                    q_clamp.reshape(-1, 1)
                                ).reshape(chunk, num_quantiles).astype(np.float64)
                                # Any remaining non-finite values (edge overflow) fall back to
                                # the raw z-norm prediction rather than a huge fill constant.
                                q_znorm = np.where(np.isfinite(q_znorm), q_znorm, q_vals.astype(np.float64))
                                accumulated_quantiles[t_start:t_end] += q_znorm
                        else:
                            accumulated_quantiles[t_start:t_end] += q_vals.astype(np.float64)

        # Average in z-norm space, then scale to original y-space.
        avg_q_znorm = (accumulated_quantiles / n_estimators).astype(np.float32)
        avg_q_raw = avg_q_znorm * self.y_std_ + self.y_mean_  # (n_test, num_quantiles)

        q_tensor = torch.from_numpy(avg_q_raw)  # (n_test, num_quantiles)
        qdist = QuantileDistribution(q_tensor, tail_type=self.tail_type)

        dtype = q_tensor.dtype

        def _decode_quantiles(qs):
            # qdist.icdf accepts (k,) alpha → returns (n_test, k)
            alpha_t = torch.tensor(list(qs), dtype=dtype)
            vals = qdist.icdf(alpha_t)  # (n_test, k)
            return [vals[:, i].numpy() for i in range(len(qs))]

        if output_type == "mean":
            return qdist.mean().numpy()
        elif output_type == "median":
            return qdist.icdf(torch.tensor(0.5, dtype=dtype)).numpy()
        elif output_type == "std":
            return qdist.variance().clamp(min=0.0).sqrt().numpy()
        elif output_type == "quantiles":
            return _decode_quantiles(quantiles)
        elif output_type in ("main", "full"):
            result: dict = {
                "mean":      qdist.mean().numpy(),
                "median":    qdist.icdf(torch.tensor(0.5, dtype=dtype)).numpy(),
                "quantiles": _decode_quantiles(quantiles),
            }
            if output_type == "full":
                result["qdist"] = qdist
                result["quantile_values"] = avg_q_raw  # (n_test, num_quantiles)
            return result
        else:
            raise ValueError(
                f"Unknown output_type {output_type!r}. "
                "Choose from: 'mean', 'median', 'std', 'quantiles', 'main', 'full'."
            )
