"""Sklearn-compatible regressor interface for the LoopICL tabular foundation model (regression).

Two decoder modes
-----------------

**Bar-distribution mode** (``LoopICLRegressor``)
  - Model: ``LoopICLReg`` — outputs ``(B, n_test, num_bars)`` logits over discrete bins
  - Ensemble: softmax → CDF-remap to common z-norm border space → average probs
  - Decode: ``BarDistribution.predict_{mean,median,quantile,mode}``

**TabICL mode** (``LoopICLTabICLRegressor``)
  - Model: ``LoopICLTabICLReg`` — outputs ``(B, n_test, num_quantiles)`` raw quantile values
  - Ensemble: inverse-transform per-estimator quantiles to z-norm space → average
  - Decode: ``QuantileDistribution.{mean,variance,...}`` with tail extrapolation

Both share the same fit / EnsembleGenerator logic; only predict() differs.

Output types (both classes)
----------------------------
``"mean"``      – E[Y]
``"median"``    – 0.5-quantile
``"quantiles"`` – list of arrays, one per quantile in ``quantiles``
``"main"``      – dict with mean, median, quantiles
``"full"``      – main + raw distribution object

Additional output types for ``LoopICLTabICLRegressor``
  ``"std"``    – standard deviation (sqrt of variance)

Usage::

    from loopicl.interface import LoopICLRegressor, LoopICLTabICLRegressor
    from loopicl.interface.wrap_model import LoopICLRegWrapper

    reg = LoopICLRegressor(wrapper=ricotta_reg_wrapper, n_estimators=8)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)                    # mean by default
    main  = reg.predict(X_test, output_type="main")

    # Pinball / quantile mode:
    from loopicl.interface.wrap_model import LoopICLRegWrapper
    tabicl_reg = LoopICLTabICLRegressor(wrapper=ricotta_tabicl_reg_wrapper, n_estimators=8)
    tabicl_reg.fit(X_train, y_train)
    preds = tabicl_reg.predict(X_test)
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
    _get_wrapper_device,
    _has_prefill,
    _quantize_cache,
    _wrapper_prefill,
    _wrapper_predict_cached,
)
from loopicl.models.bar_distribution import BarDistribution


# ---------------------------------------------------------------------------
# Output-type definitions (mirrors TabPFN v3)
# ---------------------------------------------------------------------------

OutputType = Literal["mean", "median", "mode", "quantiles", "full", "main"]

_NORM_METHODS = ["squashing_scaler_default", "quantile_uni_append"]  # TabPFN v3 regression defaults
# v3 checkpoint inference_config PREPROCESS_TRANSFORMS:
#  1. squashing_scaler_default + svd_quarter_components (no append_original)
#  2. quantile_uni + append_original='auto' (True for n_features < 500)

_Y_TRANSFORM_METHODS = [None, "safepower"]  # TabPFN v3 regression y-transform defaults


# ---------------------------------------------------------------------------
# Probability remapping across border grids (mirrors TabPFN v3's
# translate_probs_across_borders / _cdf helper)
# ---------------------------------------------------------------------------


def _translate_probs(
    probs: torch.Tensor,  # (n_test, num_bars_frm) — softmax probabilities
    frm: torch.Tensor,    # (num_bars_frm + 1,) — monotone source borders
    to: torch.Tensor,     # (num_bars_to + 1,) — monotone target borders
) -> torch.Tensor:        # (n_test, num_bars_to)
    """Redistribute probability mass from the ``frm`` grid to the ``to`` grid.

    Evaluates the CDF of the bar distribution (defined by ``probs`` on ``frm``)
    at each ``to``-border, then differences adjacent CDF values to obtain
    probabilities on the ``to`` grid.  Mirrors TabPFN v3's
    ``translate_probs_across_borders`` / ``_cdf`` helpers.
    """
    n_test, num_bars_frm = probs.shape

    # Cumulative probabilities: cumprobs_ext[:, k] = P(Y <= frm[k])
    cumprobs_ext = torch.cat(
        [torch.zeros(n_test, 1, dtype=probs.dtype), torch.cumsum(probs, dim=-1)],
        dim=-1,
    )  # (n_test, num_bars_frm + 1)

    # For each to-border, find the frm-bin it falls into
    bin_idx = (
        torch.searchsorted(frm.contiguous(), to.contiguous()) - 1
    ).clamp(0, num_bars_frm - 1)  # (num_bars_to + 1,)

    bin_widths = (frm[1:] - frm[:-1]).clamp(min=1e-10)  # (num_bars_frm,)

    # Fraction within the frm-bin (clamped to [0, 1] handles out-of-range)
    frac = ((to - frm[bin_idx]) / bin_widths[bin_idx]).clamp(0.0, 1.0)  # (num_bars_to+1,)

    cdf_left = cumprobs_ext[:, bin_idx]        # (n_test, num_bars_to + 1)
    prob_bin  = probs[:, bin_idx]              # (n_test, num_bars_to + 1)

    cdf_at_to = (cdf_left + frac.unsqueeze(0) * prob_bin).clamp(0.0, 1.0)
    cdf_at_to[:, 0]  = 0.0
    cdf_at_to[:, -1] = 1.0

    return (cdf_at_to[:, 1:] - cdf_at_to[:, :-1]).clamp_min(0.0)  # (n_test, num_bars_to)


# ---------------------------------------------------------------------------
# Shared y-transform helpers
# ---------------------------------------------------------------------------


def _build_y_transforms(
    all_y_cont: list,
    y_transform_methods: list,
    fixed_edges_1d: torch.Tensor,
) -> tuple:
    """Fit per-estimator power transforms and pre-compute border mappings.

    Parameters
    ----------
    all_y_cont         : per-estimator z-normalized y arrays
    y_transform_methods: cycling list of transform names (None or "safepower")
    fixed_edges_1d     : (num_bars+1,) model border tensor in z-norm space

    Returns
    -------
    all_y_model  : list of transformed y arrays for model input
    all_borders_t: list of (num_bars+1,) border tensors in z-norm space, or None
    all_power_tfms: list of fitted PowerTransformer objects, or None
    """
    all_y_model: list = []
    all_borders_t: list = []
    all_power_tfms: list = []

    for est_idx, y_cont in enumerate(all_y_cont):
        ytm = y_transform_methods[est_idx % len(y_transform_methods)]
        if ytm is None:
            all_y_model.append(y_cont)
            all_borders_t.append(None)
            all_power_tfms.append(None)
        elif ytm == "safepower":
            power_tfm = PowerTransformer(method="yeo-johnson", standardize=True)
            y_model = power_tfm.fit_transform(
                y_cont.reshape(-1, 1)
            ).ravel().astype(np.float32)
            # inverse_transform maps fixed_edges_1d (power-transform space) back to
            # z-norm y-space, but can fail in two ways:
            #  (a) λ > 2 or λ < 0: inverse formula raises a negative base to a
            #      non-integer power → NaN.  Example: cpu_act gives λ ≈ 2.61, so all
            #      borders below ≈ −3.15 return NaN (1426 out of 5001 values).
            #      np.sort then moves those NaN to the END, corrupting frm's high end
            #      and producing NaN probabilities in _translate_probs.
            #  (b) Near-zero λ: inverse of extreme borders (±128) ≈ ±e^128 → overflow
            #      to ±inf when cast to float32.
            # Fix: replace any non-finite result with ±1e15 keyed on the SIGN of the
            # original fixed_edges_1d value (negative edge → large-negative z-norm,
            # positive edge → large-positive z-norm), then clip for float32 safety.
            borders_t_f64 = power_tfm.inverse_transform(
                fixed_edges_1d.numpy().reshape(-1, 1)
            ).ravel()
            nan_fill = np.where(fixed_edges_1d.numpy() >= 0, 1e15, -1e15)
            borders_t_f64 = np.where(np.isfinite(borders_t_f64), borders_t_f64, nan_fill)
            borders_t_np = np.sort(
                np.clip(borders_t_f64, -1e15, 1e15).astype(np.float32)
            )
            all_y_model.append(y_model)
            all_borders_t.append(torch.from_numpy(borders_t_np))
            all_power_tfms.append(power_tfm)
        else:
            raise ValueError(
                f"Unknown y_transform_method {ytm!r}. Use None or 'safepower'."
            )

    return all_y_model, all_borders_t, all_power_tfms


# ---------------------------------------------------------------------------
# LoopICLRegressor
# ---------------------------------------------------------------------------


class LoopICLRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible regressor wrapping the LoopICL bar-distribution model.

    The model outputs a categorical distribution over ``num_bars`` bins
    whose edges are globally pre-computed and stored in ``model.fixed_edges``
    (a ``(1, num_bars+1)`` buffer on the model).  Predictions are decoded by
    computing the expected value (or other statistics) of this piecewise-
    uniform bar distribution.

    Parameters
    ----------
    wrapper : LoopICLRegWrapper
        The regression model wrapped in a ``LoopICLRegWrapper``.  Must expose
        ``wrapper._model.fixed_edges`` (shape ``(1, num_bars+1)``).
        Called as ``wrapper(X_full, y_train_idx, eval_pos)`` with numpy arrays;
        returns logits of shape ``(B, n_test, num_bars)`` for batched inputs.
    n_estimators : int, default=8
        Total number of ensemble members.
    norm_methods : list[str] or None, default=None
        X normalization variants.  ``None`` uses ``["quantile_uni", "safepower"]``.
    y_transform_methods : list or None, default=None
        Y target transform variants cycled across estimators.
        ``None`` uses ``[None, "safepower"]``.  Each estimator's y transform is
        applied after z-normalization; probabilities are remapped to a common
        z-norm border space before averaging.
    feat_shuffle_method : str, default='balanced_shuffle'
        Feature permutation strategy: ``'none'``, ``'shift'``, ``'random'``, ``'latin'``,
        ``'balanced_shuffle'``.  ``'balanced_shuffle'`` matches TabPFN v3's "shuffle" mode.
    outlier_threshold : float or None, default=None
        Z-score threshold for outlier clipping in the preprocessing pipeline.
        ``None`` disables clipping, matching TabPFN v3 regression (no outlier removal).
    fingerprint_feature : bool, default=True
        If True, append a duplicate-count column (fingerprint) to each estimator's
        input, matching TabPFN v3's ``FINGERPRINT_FEATURE=True``.
    batch_size : int or None, default=8
        Maximum ensemble members per forward call.  Reduce to save VRAM.
    test_chunk_size : int or None, default=None
        Maximum test samples per forward call. The test set is split into
        chunks of this size and processed sequentially, with each chunk
        concatenated to the full train context. Reduces peak VRAM when
        n_test is large (memory scales quadratically with sequence length).
        ``None`` processes all test samples in one call.
    cache_context : bool, default=False
        When True and the model supports ``prefill`` / ``predict_cached``, the
        train-side KV cache is built once at ``fit()`` time and reused across all
        ``predict()`` calls.  Per-estimator y-transforms and border mappings are
        also pre-computed and stored, eliminating redundant work on repeated
        ``predict()`` calls.  Automatically disables the fingerprint feature.
    quantize_kv_cache : bool, default=False
        When True, int8-quantizes the KV cache after prefill via
        ``cache.quantize()`` if supported.  Reduces peak VRAM ~4×.
    keep_cache_on_cpu : bool, default=False
        Move the KV cache to CPU immediately after prefill and restore it to
        GPU before each decode call.  More effective than the old per-batch
        behaviour when ``test_chunk_size`` is large.
    subsample_samples : int or None, default=None
        If set, each estimator draws this many training rows (stratified by
        class) instead of the full training set.  ``None`` uses all rows.
    max_features : int or None, default=None
        If set, each estimator randomly draws this many features (without
        replacement).  ``None`` uses all features.
    random_state : int or None, default=42
        Seed for reproducible ensemble generation.

    Attributes
    ----------
    n_features_in_ : int
    n_train_samples_ : int
    n_outputs_ : int  (always 1)
    X_encoder_ : XEncoder
    ensemble_generator_ : EnsembleGenerator
    context_caches_ : list  (present only when cache_context=True and model supports prefill)
    """

    def __init__(
        self,
        wrapper,
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
    ):
        self.wrapper = wrapper
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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoopICLRegressor":
        """Fit preprocessing and ensemble configuration on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)  – continuous targets

        Returns
        -------
        self
        """
        self.X_encoder_ = XEncoder()
        X_enc = self.X_encoder_.fit_transform(X)

        y_enc = np.asarray(y, dtype=np.float32).ravel()

        # Z-normalize y so the model receives values in the same distribution
        # as the prior's standardized targets (μ≈0, σ≈1).  Mirrors TabICL v2
        # (StandardScaler → fit ensemble on y_scaled) and TabPFN v3
        # (y_train_mean_ / y_train_std_ → znorm_space_bardist_).
        self.y_mean_ = float(np.mean(y_enc))
        self.y_std_ = max(float(np.std(y_enc)), 1e-9)
        y_scaled = (y_enc - self.y_mean_) / self.y_std_

        self.n_features_in_ = X_enc.shape[1]
        self.n_train_samples_ = X_enc.shape[0]
        self.n_outputs_ = 1

        norm_methods = self.norm_methods if self.norm_methods is not None else _NORM_METHODS

        # cache_context requires fingerprint=False: fingerprint depends on the
        # test set and cannot be pre-computed at fit time.
        fingerprint_feature = self.fingerprint_feature and not self.cache_context

        # No class shuffling for regression; fit on scaled y so that y_ens[i]
        # in predict() is already in standardized space (same as TabICL v2).
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

        # Build fit-time KV cache when requested and the model supports it.
        if self.cache_context and _has_prefill(self.wrapper):
            self.context_caches_, self._context_cache_batch_size_ = (
                self._build_context_cache()
            )

        return self

    # ------------------------------------------------------------------
    # _build_context_cache
    # ------------------------------------------------------------------

    def _build_context_cache(self) -> tuple:
        """Prefill the train-side KV cache and pre-compute y-transforms.

        Called at ``fit()`` time when ``cache_context=True``.  Stores per-
        estimator power-transform objects and border tensors on ``self`` so
        they need not be recomputed on every ``predict()`` call.

        Returns
        -------
        caches     : list of KV cache objects in shape-group/batch order
        batch_size : int — batch_size used so predict() can replay in order
        """
        model = object.__getattribute__(self.wrapper, "_model")
        fixed_edges: torch.Tensor = model.fixed_edges.cpu()  # (1, num_bars+1)
        fixed_edges_1d = fixed_edges.squeeze(0)

        all_X_train, all_y_cont = self.ensemble_generator_.transform_train_only()
        n_estimators = len(all_X_train)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        y_transform_methods = (
            self.y_transform_methods if self.y_transform_methods is not None
            else _Y_TRANSFORM_METHODS
        )
        all_y_model, all_borders_t, all_power_tfms = _build_y_transforms(
            all_y_cont, y_transform_methods, fixed_edges_1d
        )

        # Store pre-computed transforms so predict() doesn't redo them.
        self._cache_borders_t_ = all_borders_t
        self._cache_power_tfms_ = all_power_tfms

        # Build caches grouped by feature shape (same as predict's shape_groups).
        shape_groups: dict = defaultdict(list)
        for i, X_i in enumerate(all_X_train):
            shape_groups[X_i.shape[1]].append(i)  # group by n_features

        caches: list = []
        for _n_feats, indices in shape_groups.items():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start: start + batch_size]
                batch_X_train = np.stack(
                    [all_X_train[i] for i in batch_idx], axis=0
                )
                batch_y = np.stack([all_y_model[i] for i in batch_idx], axis=0)

                cache = _wrapper_prefill(self.wrapper, batch_X_train, batch_y)
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
        """Predict regression targets via the LoopICL bar-distribution ensemble.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)
        output_type : str, default='mean'
            One of ``'mean'``, ``'median'``, ``'mode'``, ``'quantiles'``,
            ``'main'``, ``'full'``.
        quantiles : tuple of float, default=(0.1, 0.5, 0.9)
            Quantile levels used when ``output_type`` is ``'quantiles'`` or
            ``'main'`` / ``'full'``.

        Returns
        -------
        np.ndarray
            For ``'mean'``, ``'median'``, ``'mode'``: shape ``(n_test,)``.
        list[np.ndarray]
            For ``'quantiles'``: list of ``(n_test,)`` arrays, one per quantile.
        dict
            For ``'main'`` / ``'full'``: dict with keys
            ``mean``, ``median``, ``mode``, ``quantiles``
            (plus ``bar`` for ``'full'``).
        """
        check_is_fitted(self, ["ensemble_generator_", "y_mean_", "y_std_"])

        X_enc = self.X_encoder_.transform(X)
        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_

        # Retrieve model and fixed global bin edges — move to CPU so that all
        # post-forward operations stay on CPU alongside the numpy-derived tensors.
        model = object.__getattribute__(self.wrapper, "_model")
        fixed_edges: torch.Tensor = model.fixed_edges.cpu()  # (1, num_bars+1)
        num_bars = fixed_edges.shape[1] - 1
        bar = BarDistribution(num_bars=num_bars)

        # ------------------------------------------------------------------
        # Build estimator lists
        # ------------------------------------------------------------------
        ensemble_data = self.ensemble_generator_.transform(X_enc)

        all_X: list[np.ndarray] = []
        all_y_cont: list[np.ndarray] = []  # standardized continuous y for model input

        for _norm_method, (X_ens, y_ens) in ensemble_data.items():
            for i in range(X_ens.shape[0]):
                all_X.append(X_ens[i])  # (n_total, n_features)

                # y_ens[i] is already z-normalized (fitted on y_scaled in fit()).
                # Pass it directly as continuous y — the model's reg_y_encoder
                # expects values in the prior's standardized distribution (μ≈0, σ≈1).
                all_y_cont.append(y_ens[i].astype(np.float32))  # (N,)

        n_estimators = len(all_X)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        # Resolve y-transform methods and precomputed transforms.
        y_transform_methods = (
            self.y_transform_methods if self.y_transform_methods is not None
            else _Y_TRANSFORM_METHODS
        )
        fixed_edges_1d = fixed_edges.squeeze(0)  # (num_bars+1,) in z-norm space

        use_cache_context = self.cache_context and hasattr(self, "context_caches_")

        if use_cache_context:
            # Reuse pre-computed transforms from fit() time.
            all_y_model = [y_cont for y_cont in all_y_cont]  # y is same (train-derived)
            all_borders_t = self._cache_borders_t_
        else:
            all_y_model, all_borders_t, _ = _build_y_transforms(
                all_y_cont, y_transform_methods, fixed_edges_1d
            )

        # ------------------------------------------------------------------
        # Batched forward passes — accumulate probabilities (TabPFN v3 strategy)
        # ------------------------------------------------------------------
        accumulated_probs = np.zeros((n_test, num_bars), dtype=np.float64)
        test_chunk_size = self.test_chunk_size if self.test_chunk_size is not None else n_test

        use_two_phase = _has_prefill(self.wrapper) and (
            self.test_chunk_size is not None or use_cache_context
        )

        # Group estimators by feature count so same-shape arrays can be stacked.
        shape_groups: dict = defaultdict(list)
        for i, X_i in enumerate(all_X):
            shape_groups[X_i.shape[1]].append(i)  # group by n_features

        _cache_idx = 0  # index into self.context_caches_ (used when use_cache_context)
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
                    cache = _wrapper_prefill(self.wrapper, batch_X_train, batch_y)
                    if self.quantize_kv_cache:
                        cache = _quantize_cache(cache)
                    if self.keep_cache_on_cpu:
                        cache = cache.to("cpu")

                for t_start in range(0, n_test, test_chunk_size):
                    t_end = min(t_start + test_chunk_size, n_test)

                    batch_X_chunk = np.stack(
                        [all_X[i][n_train + t_start: n_train + t_end] for i in batch_idx],
                        axis=0,
                    )

                    if use_two_phase:
                        if self.keep_cache_on_cpu:
                            device = _get_wrapper_device(self.wrapper)
                            cache = cache.to(device)
                        logits = _wrapper_predict_cached(
                            self.wrapper, batch_X_chunk, cache
                        )
                        if self.keep_cache_on_cpu:
                            cache = cache.to("cpu")
                    else:
                        batch_X = np.concatenate([batch_X_train, batch_X_chunk], axis=1)
                        logits = self.wrapper(batch_X, batch_y, eval_pos=n_train)
                        if isinstance(logits, torch.Tensor):
                            logits = logits.float().cpu().numpy()
                    # logits: (B, chunk, num_bars)

                    for j, est_idx in enumerate(batch_idx):
                        probs = torch.softmax(
                            torch.from_numpy(logits[j]).float(), dim=-1
                        )  # (chunk, num_bars)

                        bt = all_borders_t[est_idx]
                        if bt is not None:
                            # Remap from safepower-y-space to common z-norm border space
                            probs = _translate_probs(probs, frm=bt, to=fixed_edges_1d.float())
                        accumulated_probs[t_start:t_end] += probs.numpy()

        # Average probabilities, then take log so BarDistribution.predict_* can
        # re-apply softmax (exp(log(p)) ≈ p up to normalisation).
        avg_probs = (accumulated_probs / n_estimators).astype(np.float32)
        log_probs = torch.from_numpy(avg_probs).clamp(min=1e-10).log()  # (n_test, num_bars)

        # Scale fixed_edges from z-norm space → original y-space (TabPFN v3:
        #   raw_space_bardist_.borders = borders * y_train_std_ + y_train_mean_)
        edges_raw = fixed_edges * self.y_std_ + self.y_mean_    # (1, num_bars+1)
        edges_raw_t = edges_raw.expand(n_test, -1).float()      # (n_test, num_bars+1)

        # Wrap as (1, n_test, num_bars) for BarDistribution helpers
        lt = log_probs.unsqueeze(0)  # (1, n_test, num_bars)

        # ------------------------------------------------------------------
        # Decode — no inverse-transform needed; edges are already in y-space
        # ------------------------------------------------------------------
        if output_type == "mean":
            return bar.predict_mean(lt, edges_raw_t).squeeze(0).numpy()
        elif output_type == "median":
            return bar.predict_median(lt, edges_raw_t).squeeze(0).numpy()
        elif output_type == "mode":
            probs = torch.softmax(lt, dim=-1).squeeze(0)           # (n_test, num_bars)
            centers = (edges_raw[0, :-1] + edges_raw[0, 1:]) * 0.5
            return centers[probs.argmax(dim=-1)].numpy()
        elif output_type == "quantiles":
            return [
                bar.predict_quantile(lt, edges_raw_t, q).squeeze(0).numpy()
                for q in quantiles
            ]
        elif output_type in ("main", "full"):
            probs = torch.softmax(lt, dim=-1).squeeze(0)
            centers = (edges_raw[0, :-1] + edges_raw[0, 1:]) * 0.5
            result: dict = {
                "mean":      bar.predict_mean(lt, edges_raw_t).squeeze(0).numpy(),
                "median":    bar.predict_median(lt, edges_raw_t).squeeze(0).numpy(),
                "mode":      centers[probs.argmax(dim=-1)].numpy(),
                "quantiles": [
                    bar.predict_quantile(lt, edges_raw_t, q).squeeze(0).numpy()
                    for q in quantiles
                ],
            }
            if output_type == "full":
                result["bar"] = bar
            return result
        else:
            raise ValueError(
                f"Unknown output_type {output_type!r}. "
                "Choose from: 'mean', 'median', 'mode', 'quantiles', 'main', 'full'."
            )


# ---------------------------------------------------------------------------
# LoopICLTabICLRegressor — quantile / pinball decoder mode
# ---------------------------------------------------------------------------


TabICLOutputType = Literal["mean", "median", "std", "quantiles", "main", "full"]


class LoopICLTabICLRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible regressor wrapping ``LoopICLTabICLReg`` (TabICL v2 decoder style).

    Unlike ``LoopICLRegressor`` (which uses a bar-distribution discretisation),
    this class works with models that output raw quantile predictions directly
    (``num_quantiles`` evenly-spaced quantile values, trained with pinball loss).

    Ensemble strategy
    -----------------
    - Same X-normalisation / feature-shuffle / y-transform ensemble as ``LoopICLRegressor``
    - Per estimator: model outputs ``(B, n_test, num_quantiles)`` quantile values in
      the (optionally power-transformed) z-norm y-space
    - For power-transformed estimators: inverse-transform predictions back to z-norm space
    - Average raw quantile values across estimators in z-norm space
    - Scale from z-norm → original y-space (``q * y_std_ + y_mean_``)
    - Wrap in ``QuantileDistribution`` for analytical mean, variance, quantile decode

    Parameters
    ----------
    wrapper : LoopICLRegWrapper
        Wrapper around a ``LoopICLTabICLReg`` model.
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
        wrapper,
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
        self.wrapper = wrapper
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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoopICLTabICLRegressor":
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

        if self.cache_context and _has_prefill(self.wrapper):
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

                cache = _wrapper_prefill(self.wrapper, batch_X_train, batch_y)
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
        output_type: TabICLOutputType = "mean",
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

        use_two_phase = _has_prefill(self.wrapper) and (
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
                    cache = _wrapper_prefill(self.wrapper, batch_X_train, batch_y)
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
                            device = _get_wrapper_device(self.wrapper)
                            cache = cache.to(device)
                        raw = _wrapper_predict_cached(
                            self.wrapper, batch_X_chunk, cache
                        )
                        if self.keep_cache_on_cpu:
                            cache = cache.to("cpu")
                    else:
                        batch_X = np.concatenate([batch_X_train, batch_X_chunk], axis=1)
                        raw = self.wrapper(batch_X, batch_y, eval_pos=n_train)
                        if isinstance(raw, torch.Tensor):
                            raw = raw.float().cpu().numpy()
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
