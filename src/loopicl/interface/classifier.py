"""Sklearn-compatible classifier interface for the LoopICL tabular foundation model.

Ensemble strategy
-----------------
Binary classification  : probability averaging
  - balanced class permutations ([0,1]/[1,0] balanced across estimators)
  - richer normalization variants: ["none", "power", "quantile", "robust"]
  - optional stratified row subsampling per estimator
  - softmax per estimator → average probabilities

Multiclass classification: configurable via ``multiclass_strategy``
  - "logit_avg" (default): circular class-shift permutations, accumulate raw logits → single softmax
  - "prob_avg": balanced class permutations, softmax per estimator → average probabilities

Both strategies apply the class permutation to y_train before each forward
pass and apply the same permutation to un-shuffle the output logits before
accumulation.

Usage::

    from loopicl.interface import LoopICLClassifier

    clf = LoopICLClassifier()                        # default checkpoint, auto device
    clf = LoopICLClassifier(use_early_exit=True)     # EE checkpoint, exit_q=0.5
    clf = LoopICLClassifier(use_early_exit=True, exit_q=0.3)  # aggressive early exit
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

# Default checkpoints — resolved relative to the repo root.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_WEIGHTS    = str(_REPO_ROOT / "results" / "weights" / "loopicl-cls-stage2-step-30000.ckpt")
_DEFAULT_EE_WEIGHTS = str(_REPO_ROOT / "results" / "weights" / "loopicl-cls-ee-step-4000.ckpt")

from loopicl.interface._preprocessing import (
    EnsembleGenerator,
    XEncoder,
    _has_prefill,
    _quantize_cache,
    _wrapper_prefill,
    _wrapper_predict_cached,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    x = x / max(temperature, 1e-8)
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _compute_mr_bases(n_classes: int, max_classes: int) -> list:
    """Compute mixed-radix bases to cover n_classes with each base <= max_classes.

    Returns the smallest list of bases (each <= max_classes) whose product
    is >= n_classes.  Used for the many-class ECOC strategy.
    """
    import math
    if n_classes <= max_classes:
        return [n_classes]
    n_digits = math.ceil(math.log(n_classes) / math.log(max_classes))
    base = min(math.ceil(n_classes ** (1.0 / n_digits)), max_classes)
    bases = [base] * n_digits
    while int(np.prod(bases)) < n_classes:
        bases[0] = min(bases[0] + 1, max_classes)
        if int(np.prod(bases)) < n_classes:
            # need another digit
            n_digits += 1
            base = min(math.ceil(n_classes ** (1.0 / n_digits)), max_classes)
            bases = [base] * n_digits
    return bases


def _make_digit_map(n_classes: int, bases: list) -> np.ndarray:
    """Map each class index to its mixed-radix digit values.

    Returns an array of shape (n_classes, n_digits) where
    digit_map[c, d] = digit d of class c (least-significant digit first).
    """
    n_digits = len(bases)
    digit_map = np.zeros((n_classes, n_digits), dtype=np.int64)
    for c in range(n_classes):
        remainder = c
        for d in range(n_digits):
            digit_map[c, d] = remainder % bases[d]
            remainder //= bases[d]
    return digit_map


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Binary defaults (probability averaging)
_BINARY_NORM_METHODS = ["none", "power", "quantile", "robust"]
_BINARY_CLASS_SHUFFLE = "balanced_shuffle"

# Multiclass defaults (logit averaging)
_MULTI_NORM_METHODS = ["none", "power"]
_MULTI_CLASS_SHUFFLE = "shift"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class LoopICLClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier wrapping the LoopICL tabular foundation model.

    Binary vs. multiclass behaviour is selected automatically at ``fit`` time.

    Binary
    ~~~~~~
    - Class permutations: ``balanced_shuffle`` — all unique permutations
      ([0,1] and [1,0]) distributed evenly across estimators.
    - Normalization: ``["none", "power", "quantile", "robust"]``.
    - Aggregation: softmax per estimator, then average probabilities.

    Multiclass
    ~~~~~~~~~~
    Controlled by ``multiclass_strategy``:

    - ``"logit_avg"`` *(default)*: ``shift`` class permutations, ``["none", "power"]``
      normalization, accumulate raw logits → single softmax.
    - ``"prob_avg"``: ``balanced_shuffle`` class permutations,
      ``["none", "power", "quantile", "robust"]`` normalization,
      softmax per estimator → average probabilities.

    Parameters
    ----------
    model : LoopICL or None, default=None
        The LoopICL model.  If ``None``, loaded automatically from ``weights_path``.
    weights_path : str or None, default=None
        Path to a ``.ckpt`` checkpoint containing ``"config"`` and
        ``"ema_state_dict"`` keys.  Used only when ``model`` is ``None``.
        Defaults to the standard or early-exit checkpoint depending on
        ``use_early_exit``.
    use_early_exit : bool, default=False
        When ``True``, loads the early-exit checkpoint
        (``loopicl-cls-ee-step-4000.ckpt``) and sets ``exit_q=0.5`` if no
        explicit ``exit_q`` is given.  The early-exit model gates each ICL
        loop and exits early once the gate CDF exceeds ``exit_q``, trading
        a small accuracy cost for significant inference speedup.
    device : str or None, default=None
        Torch device to run inference on (e.g. ``"cuda"``, ``"cpu"``).
        If ``None``, auto-detected from the model parameters.
    dtype : str, default="float32"
        Compute dtype: ``"float32"``, ``"float16"``, or ``"bfloat16"``.
    num_loops : int or None, default=None
        Override the number of ICL loop iterations at inference time.
        If ``None``, uses the model's default.
    mrl_dim : int or None, default=None
        MRL dimension override.
    loop_residual_scaling_fn : str or None, default=None
        Override the loop residual scaling function.
    exit_q : float or None, default=None
        Early-exit quantile threshold.
    n_estimators : int, default=8
        Total number of ensemble members.
    norm_methods : str or list[str] or None, default=None
        Override the normalization methods. ``None`` auto-selects based on
        binary vs. multiclass (see above).
    feat_shuffle_method : str, default='latin'
        Feature permutation strategy: ``'none'``, ``'shift'``, ``'random'``,
        ``'latin'``.
    class_shuffle_method : str or None, default=None
        Override the class-label permutation strategy. ``None`` auto-selects
        (``'balanced_shuffle'`` for binary, ``'shift'`` for multiclass).
        Options: ``'none'``, ``'shift'``, ``'latin'``, ``'random'``,
        ``'balanced_shuffle'``.
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier clipping in the preprocessing pipeline.
    softmax_temperature : float, default=0.9
        Temperature for the softmax: ``softmax(logits / T)``.
    batch_size : int or None, default=8
        Maximum ensemble members per forward call. Reduce to save VRAM.
        ``None`` processes all members in one call.
    test_chunk_size : int or None, default=None
        Maximum test samples per forward call. The test set is split into
        chunks of this size and processed sequentially, with each chunk
        concatenated to the full train context. Reduces peak VRAM when
        n_test is large (memory scales quadratically with sequence length).
        ``None`` processes all test samples in one call.
    cache_context : bool, default=False
        When True and the wrapped model supports ``prefill`` / ``predict_cached``,
        the train-side KV cache is built once at ``fit()`` time and reused
        across all ``predict_proba()`` calls.  Eliminates redundant prefill
        computation when calling ``predict_proba()`` multiple times on the
        same fitted model.  Automatically disables the fingerprint feature
        (which requires test data and cannot be pre-computed at fit time).
    quantize_kv_cache : bool, default=False
        When True, int8-quantizes the KV cache immediately after prefill via
        ``cache.quantize()`` if the cache object supports it.  Reduces peak
        VRAM ~4× when holding multiple ensemble caches simultaneously, at a
        small accuracy cost.
    keep_cache_on_cpu : bool, default=False
        When the wrapped model supports prefill/decode, moves the KV cache to
        CPU immediately after prefill and restores it to GPU before each decode
        call.  Reduces peak VRAM at the cost of extra host↔device transfers.
        More effective than the old per-batch behaviour when ``test_chunk_size``
        is large.
    subsample_samples : int or None, default=None
        If set, each estimator draws this many training rows (stratified by
        class) instead of the full training set. Useful when the dataset is
        too large to fit in GPU memory. ``None`` (default) uses all rows.
    max_features : int or None, default=None
        If set, each estimator randomly draws this many features (without
        replacement) from the full feature set. Different
        estimators draw different subsets; averaging over the ensemble recovers
        coverage of all features. Useful for high-dimensional datasets that
        exceed GPU memory due to column-attention cost. ``None`` uses all features.
    multiclass_strategy : str, default="logit_avg"
        Aggregation strategy for multiclass problems. ``"logit_avg"`` accumulates
        raw logits and applies a single softmax at the end.
        ``"prob_avg"`` applies softmax per estimator and averages probabilities,
        and also switches to ``balanced_shuffle`` class permutations and the
        richer normalization set used for binary.
        Ignored for binary classification.
    random_state : int or None, default=42
        Seed for reproducible ensemble generation.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_classes_ : int
    n_features_in_ : int
    y_encoder_ : LabelEncoder
    ensemble_generator_ : EnsembleGenerator
    context_caches_ : dict  (present only when cache_context=True and model supports prefill)
    """

    def __init__(
        self,
        model=None,
        weights_path: Optional[str] = None,
        use_early_exit: bool = False,
        device=None,
        dtype: str = "float32",
        num_loops: Optional[int] = None,
        mrl_dim: Optional[int] = None,
        loop_residual_scaling_fn: Optional[str] = None,
        exit_q: Optional[float] = None,
        n_estimators: int = 8,
        norm_methods: Optional[str | list] = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: Optional[str] = None,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        batch_size: Optional[int] = 8,
        test_chunk_size: Optional[int] = None,
        cache_context: bool = False,
        quantize_kv_cache: bool = False,
        keep_cache_on_cpu: bool = False,
        subsample_samples: Optional[int] = None,
        max_features: Optional[int] = None,
        multiclass_strategy: str = "prob_avg",
        random_state: Optional[int] = 42,
    ):
        if model is None:
            if weights_path is None:
                weights_path = _DEFAULT_EE_WEIGHTS if use_early_exit else _DEFAULT_WEIGHTS
            model = self._load_checkpoint(weights_path, device)
            if use_early_exit and exit_q is None:
                exit_q = 0.5
        self.model = model
        self.weights_path = weights_path
        self.use_early_exit = use_early_exit
        self.device = device
        self.dtype = dtype
        self.num_loops = num_loops
        self.mrl_dim = mrl_dim
        self.loop_residual_scaling_fn = loop_residual_scaling_fn
        self.exit_q = exit_q
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.batch_size = batch_size
        self.test_chunk_size = test_chunk_size
        self.cache_context = cache_context
        self.quantize_kv_cache = quantize_kv_cache
        self.keep_cache_on_cpu = keep_cache_on_cpu
        self.subsample_samples = subsample_samples
        self.max_features = max_features
        self.multiclass_strategy = multiclass_strategy
        self.random_state = random_state

    @staticmethod
    def _load_checkpoint(weights_path: str, device=None):
        import dataclasses
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        # Rename legacy key
        if "max_classes" in cfg and "max_num_classes" not in cfg:
            cfg["max_num_classes"] = cfg.pop("max_classes")

        # Detect early-exit checkpoint by presence of gate config keys
        if "gate_delta_out_dim" in cfg:
            from loopicl.models.loopicl.model_early_exit import (
                LoopICLEarlyExit,
                LoopICLEarlyExitConfig,
            )
            valid = set(LoopICLEarlyExitConfig.__dataclass_fields__)
            cfg = {k: v for k, v in cfg.items() if k in valid}
            model = LoopICLEarlyExit(**cfg)
        else:
            from loopicl.models.loopicl.model import LoopICL, LoopICLConfig
            valid = {f.name for f in dataclasses.fields(LoopICLConfig)}
            cfg = {k: v for k, v in cfg.items() if k in valid}
            model = LoopICL(LoopICLConfig(**cfg))

        state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_key])
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)
        model.eval()
        return model

    def _resolve_device(self):
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _call_model(self, X_np: np.ndarray, y_np: np.ndarray, eval_pos: int) -> np.ndarray:
        """Convert numpy arrays to tensors, run the model, return numpy logits.

        Supports both batched (3-D X) and single (2-D X) inputs.
        """
        device = self._resolve_device()
        X_t = torch.as_tensor(X_np.astype(np.float32), device=device)
        y_t = torch.as_tensor(y_np.astype(np.float32), device=device)

        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(self.dtype)

        kwargs: dict = {"return_logits": True}
        if self.num_loops is not None:
            kwargs["num_loops"] = self.num_loops
        if self.mrl_dim is not None:
            kwargs["mrl_dim"] = self.mrl_dim
        if self.exit_q is not None:
            kwargs["exit_q"] = self.exit_q

        orig_fn = None
        if self.loop_residual_scaling_fn is not None and hasattr(self.model, "loop_residual_scaling_fn"):
            orig_fn = self.model.loop_residual_scaling_fn
            self.model.loop_residual_scaling_fn = self.loop_residual_scaling_fn

        try:
            with torch.no_grad():
                if amp_dtype is not None and "cuda" in str(device):
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        if X_t.ndim == 3:
                            out = self.model(X_t, y_t, **kwargs)
                        else:
                            out = self.model(X_t.unsqueeze(0), y_t.unsqueeze(0), **kwargs).squeeze(0)
                else:
                    if X_t.ndim == 3:
                        out = self.model(X_t, y_t, **kwargs)
                    else:
                        out = self.model(X_t.unsqueeze(0), y_t.unsqueeze(0), **kwargs).squeeze(0)
        finally:
            if orig_fn is not None:
                self.model.loop_residual_scaling_fn = orig_fn

        return out.float().cpu().numpy()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoopICLClassifier":
        """Prepare the classifier for inference.

        Fits preprocessing and ensemble configurations on the training data.
        Binary vs. multiclass defaults are resolved here.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_encoder_ = XEncoder()
        X_enc = self.X_encoder_.fit_transform(X)

        self.y_encoder_ = LabelEncoder()
        y_enc = self.y_encoder_.fit_transform(y).astype(np.float32)

        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_enc.shape[1]
        self._n_train = X_enc.shape[0]

        # Many-class support: if n_classes exceeds the model's capacity, use
        # mixed-radix decomposition.
        _max_classes = getattr(self.model, "max_classes", None)
        if _max_classes is not None and self.n_classes_ > _max_classes:
            self._mr_bases = _compute_mr_bases(self.n_classes_, _max_classes)
            self._mr_digit_map = _make_digit_map(self.n_classes_, self._mr_bases)
        else:
            self._mr_bases = None
            self._mr_digit_map = None

        is_binary = self.n_classes_ == 2
        use_prob_avg = is_binary or self.multiclass_strategy == "prob_avg"

        # Resolve norm_methods and class_shuffle_method
        norm_methods = self.norm_methods
        class_shuffle_method = self.class_shuffle_method

        if norm_methods is None:
            norm_methods = _BINARY_NORM_METHODS if use_prob_avg else _MULTI_NORM_METHODS

        if class_shuffle_method is None:
            class_shuffle_method = _BINARY_CLASS_SHUFFLE if use_prob_avg else _MULTI_CLASS_SHUFFLE

        # cache_context requires fingerprint=False: fingerprint depends on the
        # test set (via salt = n_total_rows * n_features) and cannot be
        # pre-computed at fit time.
        fingerprint_feature = not self.cache_context

        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            subsample_samples=self.subsample_samples,
            max_features=self.max_features,
            fingerprint_feature=fingerprint_feature,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X_enc, y_enc)

        # Build fit-time KV cache when requested and the model supports it.
        if self.cache_context and _has_prefill(self.model):
            self.context_caches_, self._context_cache_batch_size_ = (
                self._build_context_cache()
            )

        return self

    # ------------------------------------------------------------------
    # _build_context_cache
    # ------------------------------------------------------------------

    def _build_context_cache(self) -> tuple:
        """Prefill the train-side KV cache for every estimator batch.

        Called at ``fit()`` time when ``cache_context=True``.  Returns
        ``(caches, batch_size_used)`` where ``caches`` is a dict mapping
        estimator-batch start indices to their KV cache objects.
        """
        all_X_train, all_y = self.ensemble_generator_.transform_train_only()
        n_estimators = len(all_X_train)
        batch_size = self.batch_size if self.batch_size is not None else n_estimators

        caches: dict = {}
        for start in range(0, n_estimators, batch_size):
            end = min(start + batch_size, n_estimators)
            batch_X_train = np.stack(all_X_train[start:end], axis=0)
            batch_y = np.stack(all_y[start:end], axis=0)

            cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
            if self.quantize_kv_cache:
                cache = _quantize_cache(cache)
            if self.keep_cache_on_cpu:
                cache = cache.to("cpu")
            caches[start] = cache

        return caches, batch_size

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray, num_loops: Optional[int] = None) -> np.ndarray:
        """Predict class probabilities via the LoopICL ensemble.

        Parameters
        ----------
        X : array-like of shape (n_test, n_features)
        num_loops : int or None, default=None
            Override the number of ICL loop iterations for this call only.
            If ``None``, uses the value set at construction time.

        Returns
        -------
        proba : ndarray of shape (n_test, n_classes)
        """
        _prev_num_loops = self.num_loops
        if num_loops is not None:
            self.num_loops = num_loops
        try:
            return self._predict_proba_impl(X)
        finally:
            self.num_loops = _prev_num_loops

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["ensemble_generator_"])

        X_enc = self.X_encoder_.transform(X)

        if getattr(self, "_mr_bases", None) is not None:
            return self._predict_proba_mixed_radix(X_enc)

        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_
        is_binary = self.n_classes_ == 2
        use_prob_avg = is_binary or self.multiclass_strategy == "prob_avg"

        ensemble_data = self.ensemble_generator_.transform(X_enc)

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_shuffles: list[np.ndarray] = []

        for norm_method, (X_ens, y_ens) in ensemble_data.items():
            shuffles = self.ensemble_generator_.class_shuffles_[norm_method]
            for i in range(X_ens.shape[0]):
                all_X.append(X_ens[i])
                all_y.append(y_ens[i])
                all_shuffles.append(np.asarray(shuffles[i], dtype=np.intp))

        n_estimators = len(all_X)
        accumulated = np.zeros((n_test, self.n_classes_), dtype=np.float32)
        test_chunk_size = self.test_chunk_size if self.test_chunk_size is not None else n_test

        # Determine which decode path to use:
        #   use_cache_context  — use fit-time KV cache (cache_context=True)
        #   use_two_phase      — call prefill+decode per predict() call
        #   standard path      — single concatenated forward pass
        use_cache_context = self.cache_context and hasattr(self, "context_caches_")
        use_two_phase = _has_prefill(self.model) and (
            self.test_chunk_size is not None or use_cache_context
        )

        # Use fit-time batch_size when replaying stored caches so indices match.
        batch_size = (
            self._context_cache_batch_size_
            if use_cache_context
            else (self.batch_size if self.batch_size is not None else n_estimators)
        )

        for start in range(0, n_estimators, batch_size):
            end = min(start + batch_size, n_estimators)

            if use_cache_context:
                # Reuse the cache built at fit() time — no prefill needed.
                cache = self.context_caches_[start]
            elif use_two_phase:
                batch_X_train = np.stack(
                    [x[:n_train] for x in all_X[start:end]], axis=0
                )
                batch_y = np.stack(all_y[start:end], axis=0)
                cache = _wrapper_prefill(self.model, self._resolve_device(), self.dtype, batch_X_train, batch_y)
                if self.quantize_kv_cache:
                    cache = _quantize_cache(cache)
                # Move to CPU immediately after prefill so the test-chunk
                # forward pass doesn't compete for the same VRAM.
                if self.keep_cache_on_cpu:
                    cache = cache.to("cpu")
            else:
                batch_X_train = np.stack(
                    [x[:n_train] for x in all_X[start:end]], axis=0
                )
                batch_y = np.stack(all_y[start:end], axis=0)

            for t_start in range(0, n_test, test_chunk_size):
                t_end = min(t_start + test_chunk_size, n_test)

                batch_X_chunk = np.stack(
                    [x[n_train + t_start: n_train + t_end] for x in all_X[start:end]],
                    axis=0,
                )

                if use_two_phase:
                    # Restore cache to GPU only for this decode call, then
                    # immediately push back to CPU to free VRAM for the forward pass.
                    if self.keep_cache_on_cpu:
                        device = self._resolve_device()
                        cache = cache.to(device)
                    logits = _wrapper_predict_cached(
                        self.model, self._resolve_device(), self.dtype, batch_X_chunk, cache
                    )
                    if self.keep_cache_on_cpu:
                        cache = cache.to("cpu")
                else:
                    batch_X = np.concatenate(
                        [batch_X_train, batch_X_chunk], axis=1
                    )
                    logits = self._call_model(batch_X, batch_y, eval_pos=n_train)
                # logits: (B, chunk, n_classes) in permuted class order

                for i in range(end - start):
                    perm = all_shuffles[start + i]
                    # Un-permute: model was trained with class perm[k] at slot k,
                    # so logit[:,k] is the score for original class perm[k].
                    # out[..., perm] maps slot k → original class position perm[k].
                    out = logits[i][..., perm]  # (chunk, n_classes)

                    if use_prob_avg:
                        # Probability averaging: softmax per estimator, average probabilities
                        accumulated[t_start:t_end] += _softmax(
                            out, temperature=self.softmax_temperature
                        )
                    else:
                        # Logit averaging: accumulate raw logits, softmax at the end
                        accumulated[t_start:t_end] += out

        accumulated /= n_estimators

        if not use_prob_avg:
            accumulated = _softmax(accumulated, temperature=self.softmax_temperature)

        # Normalize in float64 to avoid float32 precision drift that causes
        # sklearn's brier_score to warn "y_prob values do not sum to one".
        accum64 = accumulated.astype(np.float64)
        accum64 /= accum64.sum(axis=1, keepdims=True)
        return accum64.astype(np.float32)

    # ------------------------------------------------------------------
    # _predict_proba_mixed_radix
    # ------------------------------------------------------------------

    def _predict_proba_mixed_radix(self, X_enc: np.ndarray) -> np.ndarray:
        """predict_proba for datasets with more classes than the model supports.

        Decomposes the K-class problem into len(_mr_bases) sub-problems using
        mixed-radix encoding: class c → (digit_0, digit_1, ...) where each
        digit_d = (c // prod(bases[:d])) % bases[d].

        For each digit position d (a bases[d]-class problem), the model is run
        with relabeled training labels and produces per-digit probabilities.
        Final class probabilities are obtained by multiplying per-digit probs:
            P(class c) ∝ ∏_d P(digit_d = digit_map[c, d])

        Uses a mixed-radix decomposition applied at the interface level rather
        than inside the model.
        """
        import torch

        n_test = X_enc.shape[0]
        n_train = self.ensemble_generator_.effective_n_train_
        batch_size = self.batch_size if self.batch_size is not None else 1

        # Collect all ensemble views (X with feature/norm diversity)
        ensemble_data = self.ensemble_generator_.transform(X_enc)

        # Pre-collect per-estimator arrays and their inverse class shuffles
        all_X = []       # list of (n_total, n_features)
        all_y_orig = []  # list of (n_train,) — original class labels (0..n_classes-1)

        for norm_method, (X_ens, y_ens) in ensemble_data.items():
            shuffles = self.ensemble_generator_.class_shuffles_[norm_method]
            for i, perm in enumerate(shuffles):
                all_X.append(X_ens[i])
                if self.ensemble_generator_.class_shuffle_method == "none":
                    # y passed through as-is (e.g. regression mode; shouldn't
                    # reach here, but handle gracefully)
                    all_y_orig.append(y_ens[i].astype(np.int64))
                else:
                    inv_perm = np.argsort(perm)
                    all_y_orig.append(inv_perm[y_ens[i].astype(np.intp)])

        n_estimators = len(all_X)

        # Accumulate log-probabilities across digit positions
        log_proba = np.zeros((n_test, self.n_classes_), dtype=np.float64)

        for d, base_d in enumerate(self._mr_bases):
            digit_map_d = self._mr_digit_map[:, d]  # (n_classes,): class → digit value

            accum_d = np.zeros((n_test, base_d), dtype=np.float64)

            for start in range(0, n_estimators, batch_size):
                end = min(start + batch_size, n_estimators)
                bs = end - start

                X_batch = np.stack(all_X[start:end], axis=0)  # (bs, n_total, n_feat)
                y_digit_batch = np.stack(
                    [digit_map_d[all_y_orig[j]] for j in range(start, end)],
                    axis=0,
                ).astype(np.float32)  # (bs, n_train)

                logits = self._call_model(X_batch, y_digit_batch, eval_pos=n_train)

                # logits: (bs, n_test, max_classes) — take first base_d slots
                accum_d += logits[:, :, :base_d].astype(np.float64).sum(axis=0)

            accum_d /= n_estimators  # average logits

            # Log-softmax over digit classes
            accum_d -= accum_d.max(axis=-1, keepdims=True)
            log_probs_d = accum_d - np.log(np.exp(accum_d).sum(axis=-1, keepdims=True))

            # Add log P(digit_d = digit_map[c, d]) for every class c
            log_proba += log_probs_d[:, digit_map_d]  # (n_test, n_classes)

        # Convert to probabilities and normalise (keep float64 through division
        # to avoid float32 precision drift that causes sklearn's brier_score
        # to warn "y_prob values do not sum to one").
        log_proba -= log_proba.max(axis=-1, keepdims=True)
        proba = np.exp(log_proba)          # float64
        proba /= proba.sum(axis=-1, keepdims=True)
        return proba.astype(np.float32)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, num_loops: Optional[int] = None) -> np.ndarray:
        """Predict class labels for X."""
        proba = self.predict_proba(X, num_loops=num_loops)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
