"""High-level interface wrapper for cross-model comparison experiments.

``wrap_interface`` loads an external interface (TabPFN, TabICL, LimiX) and
replaces its internal model with a provided ``TransformerWrapper``, allowing
you to use a different transformer backbone through that interface's
preprocessing and ensembling pipeline.

Example::

    from loopicl.interface.wrap_interface import wrap_interface, TransformerWrapper

    # tabpfn_v2 interface powered by a custom TransformerWrapper subclass
    clf = wrap_interface("tabpfn_v2", task_type="cls", model=my_wrapper)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import numpy as np
import torch


class TransformerWrapper:
    """Base class for wrapping external transformer models (TabPFN, LimiX, etc.).

    Provides a unified ``wrapper(X, y_train, eval_pos)`` interface —
    numpy arrays in, logit tensor out — for models used in cross-model
    comparison experiments.

    Subclasses implement ``_forward(X_tensor, y_tensor, eval_pos)``.
    """

    def __init__(self, model, device, dtype: str = "float32"):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_dtype", dtype)

    def __call__(self, X: np.ndarray, y_train: np.ndarray, eval_pos: int) -> torch.Tensor:
        device = object.__getattribute__(self, "_device")
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y_train, dtype=np.float32)
        X_t = torch.as_tensor(X_np, device=device)
        y_t = torch.as_tensor(y_np, device=device)
        dtype = object.__getattribute__(self, "_dtype")
        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(dtype)
        with torch.no_grad():
            if amp_dtype is not None and "cuda" in str(device):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    return self._forward(X_t, y_t, eval_pos)
            return self._forward(X_t, y_t, eval_pos)

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        raise NotImplementedError

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_model"), name)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        return f"{type(self).__name__}({model!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FrozenByteSize:
    """Picklable replacement for ``lambda: _byte_size``.

    TabPFN's ``fit()`` calls ``_initialize_model_variables()`` to reload the
    weights and reset ``models_``.  We patch that method with this object so
    the call is a no-op (returns the pre-computed byte_size) and our custom
    adapter in ``models_[0]`` is never overwritten.
    """

    def __init__(self, byte_size: int) -> None:
        self._byte_size = byte_size

    def __call__(self) -> int:
        return self._byte_size


# ---------------------------------------------------------------------------
# Model adapters  (TransformerWrapper → each interface's internal convention)
# ---------------------------------------------------------------------------

class _TabPFNModelAdapter:
    """Adapts a TransformerWrapper to TabPFN's internal model calling convention.

    TabPFN internally calls ``model(X, y_train, **kwargs)`` where:
        X        : ``(n_total, 1, n_features)``  — sequence-dim in position 1
        y_train  : ``(n_train,)``
        returns  : ``(n_test, 1, n_classes)``    — sequence-dim preserved

    Extra kwargs (``only_return_standard_out``, ``categorical_inds``,
    ``performance_options``, ``task_type``) are passed by the installed tabpfn
    inference engine and are silently ignored here.

    This adapter strips the sequence dim, calls ``wrapper._forward``, and adds
    the dim back so TabPFNClassifier's pipeline keeps working unchanged.
    """

    def __init__(self, wrapper: TransformerWrapper):
        self._wrapper = wrapper

    def __call__(self, X: torch.Tensor, y_train: torch.Tensor, **kwargs) -> torch.Tensor:
        # X: (n_total, 1, n_features) → (n_total, n_features)
        if X.ndim == 3 and X.shape[1] == 1:
            X = X.squeeze(1)
        eval_pos = y_train.shape[0]
        out = self._wrapper._forward(X, y_train, eval_pos)  # (n_test, n_classes)
        return out.unsqueeze(1)                              # (n_test, 1, n_classes)

    def get_default_performance_options(self):
        """Required by the tabpfn inference engine before each forward call."""
        from tabpfn.architectures.interface import PerformanceOptions
        return PerformanceOptions()

    # TabPFNClassifier may call these on its model
    def eval(self):   return self
    def train(self):  return self
    def to(self, *a, **kw): return self
    def forward(self, X: torch.Tensor, y_train: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(X, y_train, **kwargs)

    def parameters(self):
        # TabPFN v3 calls model.parameters() to infer device via _get_current_device.
        # Yield a dummy parameter on the correct device so _get_current_device works.
        device = object.__getattribute__(self._wrapper, "_device")
        yield torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)


class _TabICLModelAdapter:
    """Adapts a TransformerWrapper to TabICL's internal model calling convention.

    TabICL internally calls ``model(X, y_train, return_logits=True)`` where:
        X        : ``(batch, n_total, n_features)``
        y_train  : ``(batch, n_train)``
        returns  : ``(batch, n_test, n_classes)``
    """

    def __init__(self, wrapper: TransformerWrapper):
        self._wrapper = wrapper
        self.max_classes = 10

    def __call__(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        return_logits: bool = True,
        feature_shuffles=None,
        softmax_temperature=None,
        inference_config=None,
    ) -> torch.Tensor:
        # X: (batch, n_total, n_features), y_train: (batch, n_train)
        eval_pos = y_train.shape[1]
        # Use max label + 1 so all classes 0..K-1 are included even if some
        # are absent from the first batch's training slice.
        num_classes = int(y_train.max().item()) + 1 if self.max_classes > 0 else 0

        # Apply autocast matching the wrapper's dtype (e.g. bfloat16), mirroring
        # what TransformerWrapper.__call__ does — _forward is called directly here
        # (inputs are already tensors) so we must apply it manually.
        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(object.__getattribute__(self._wrapper, "_dtype"))
        if amp_dtype is not None and X.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = self._wrapper._forward(X, y_train, eval_pos)[:, :, :num_classes]
        else:
            out = self._wrapper._forward(X, y_train, eval_pos)[:, :, :num_classes]

        if not return_logits:
            out = torch.softmax(out / softmax_temperature, dim=-1)
        return out

    def predict_stats(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        output_type: "str | list[str]" = "mean",
        alphas=None,
        **kwargs,
    ) -> torch.Tensor:
        """Regression stats delegate — called by TabICLRegressor instead of __call__."""
        eval_pos = y_train.shape[1]
        out = self._wrapper._forward(X, y_train, eval_pos)  # (batch, n_test, num_outputs)

        output_type_list = [output_type] if isinstance(output_type, str) else list(output_type)
        results = {}
        if "mean" in output_type_list:
            results["mean"] = out.mean(dim=-1) if out.dim() > 2 else out
        if "variance" in output_type_list:
            results["variance"] = out.var(dim=-1) if out.dim() > 2 else torch.zeros_like(out)
        if "median" in output_type_list:
            results["median"] = out.median(dim=-1).values if out.dim() > 2 else out
        if "quantiles" in output_type_list:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            sorted_out = out.sort(dim=-1).values if out.dim() > 2 else out.unsqueeze(-1)
            n_q = sorted_out.shape[-1]
            indices = [min(round(a * n_q), n_q - 1) for a in alphas]
            results["quantiles"] = sorted_out[..., indices]

        return results[output_type_list[0]] if len(output_type_list) == 1 else results

    def eval(self):   return self
    def train(self):  return self
    def to(self, *a, **kw): return self
    def parameters(self):   return iter([])


class _LimiXModelAdapter:
    """Adapts a TransformerWrapper to LimiX's internal model calling convention.

    LimiX internally calls
        ``model(X, y_full, eval_pos=eval_pos, task_type=task_type)``
    where:
        X      : ``(batch, n_total, n_features)``
        y_full : ``(batch, n_total)``  — test positions are zero-padded
        returns: ``(batch, n_test, n_classes)``
    """

    def __init__(self, wrapper: TransformerWrapper):
        self._wrapper = wrapper

    def __call__(
        self,
        X: torch.Tensor,
        y_full: torch.Tensor,
        eval_pos: int = 0,
        task_type: str = "cls",
    ) -> torch.Tensor:
        # y_full has zeros at test positions; slice to get y_train
        y_train = y_full[:, :eval_pos]  # (batch, n_train)
        return self._wrapper._forward(X, y_train, eval_pos)  # (batch, n_test, ...)

    def eval(self):   return self
    def train(self):  return self
    def to(self, *a, **kw): return self
    def parameters(self):   return iter([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(*args, **kwargs):
    """No-op placeholder for methods that must be suppressed after model injection."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_interface(
    interface_name: str,
    task_type: str = "cls",
    model: TransformerWrapper | None = None,
    **kwargs,
):
    """Load a high-level interface and replace its model with a TransformerWrapper.

    Args:
        interface_name: Whose interface (preprocessing + API) to use.  One of:
                        ``"tabpfn_v2"``, ``"tabpfn_v2_5"``, ``"tabpfn_v2_6"``,
                        ``"tabpfn_v3"``,
                        ``"tabicl"``, ``"tabicl_v2"``, ``"tabicl_v1"``,
                        ``"tabicl_v1_1"``, ``"limix"``.
        task_type:      ``"cls"`` (classification) or ``"reg"`` (regression).
        model:          A ``TransformerWrapper`` subclass instance — the transformer
                        that will be used for inference instead of the interface's
                        default model.
        **kwargs:       Extra keyword arguments forwarded to the estimator constructor
                        (e.g. ``device``, ``batch_size``).

    Returns:
        The interface object (e.g. ``TabPFNClassifier``) with its internal model
        replaced by the provided ``TransformerWrapper``.
    """
    if model is None:
        raise ValueError("model must be a TransformerWrapper instance.")
    if not isinstance(model, TransformerWrapper):
        raise TypeError(
            f"model must be a TransformerWrapper, got {type(model).__name__}."
        )

    key = interface_name.lower()

    # ------------------------------------------------------------------
    # TabPFN v2
    # ------------------------------------------------------------------
    if key in ("tabpfn_v2",):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = (
            TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            if task_type == "cls" else
            TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        )
        _byte_size = est._initialize_model_variables()
        est.models_[0] = _TabPFNModelAdapter(model)
        est._initialize_model_variables = _FrozenByteSize(_byte_size)
        return est

    # ------------------------------------------------------------------
    # TabPFN v2.5
    # ------------------------------------------------------------------
    if key in ("tabpfn_v2_5", "tabpfn_2_5"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = (
            TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
            if task_type == "cls" else
            TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
        )
        _byte_size = est._initialize_model_variables()
        est.models_[0] = _TabPFNModelAdapter(model)
        est._initialize_model_variables = _FrozenByteSize(_byte_size)
        return est

    # ------------------------------------------------------------------
    # TabPFN v3
    # ------------------------------------------------------------------
    if key in ("tabpfn_v3", "tabpfn_3"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = (
            TabPFNClassifier.create_default_for_version(ModelVersion.V3)
            if task_type == "cls" else
            TabPFNRegressor.create_default_for_version(ModelVersion.V3)
        )
        _byte_size = est._initialize_model_variables()
        est.models_[0] = _TabPFNModelAdapter(model)
        est._initialize_model_variables = _FrozenByteSize(_byte_size)
        return est

    # ------------------------------------------------------------------
    # TabPFN v2.6
    # ------------------------------------------------------------------
    if key in ("tabpfn_v2_6", "tabpfn_2_6"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = (
            TabPFNClassifier.create_default_for_version(ModelVersion.V2_6)
            if task_type == "cls" else
            TabPFNRegressor.create_default_for_version(ModelVersion.V2_6)
        )
        _byte_size = est._initialize_model_variables()
        est.models_[0] = _TabPFNModelAdapter(model)
        est._initialize_model_variables = _FrozenByteSize(_byte_size)
        return est

    # ------------------------------------------------------------------
    # TabICL v1
    # ------------------------------------------------------------------
    if key in ("tabicl_v1", "tabicl_1"):
        if task_type != "cls":
            raise ValueError("TabICL v1 only supports classification.")
        from tabicl import TabICLClassifier
        est = TabICLClassifier(checkpoint_version="tabicl-classifier-v1-20250208.ckpt", **kwargs)
        est._resolve_device()
        est._load_model()
        est.model_ = _TabICLModelAdapter(model)
        est._load_model = _noop
        return est

    # ------------------------------------------------------------------
    # TabICL v1.1
    # ------------------------------------------------------------------
    if key in ("tabicl_v1_1", "tabicl_1_1"):
        if task_type != "cls":
            raise ValueError("TabICL v1.1 only supports classification.")
        from tabicl import TabICLClassifier
        est = TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-20250506.ckpt", **kwargs)
        est._resolve_device()
        est._load_model()
        est.model_ = _TabICLModelAdapter(model)
        est._load_model = _noop
        return est

    # ------------------------------------------------------------------
    # TabICL v2
    # ------------------------------------------------------------------
    if key in ("tabicl", "tabicl_v2", "tabicl_2"):
        from tabicl import TabICLClassifier, TabICLRegressor
        est = (
            TabICLClassifier(checkpoint_version="tabicl-classifier-v2-20260212.ckpt", **kwargs)
            if task_type == "cls" else
            TabICLRegressor(checkpoint_version="tabicl-regressor-v2-20260212.ckpt", **kwargs)
        )
        est._resolve_device()
        est._load_model()
        est.model_ = _TabICLModelAdapter(model)
        est._load_model = _noop
        return est

    # ------------------------------------------------------------------
    # LimiX
    # ------------------------------------------------------------------
    if key == "limix":
        from limix import LimiXClassifier, LimiXRegressor
        est = LimiXClassifier(**kwargs) if task_type == "cls" else LimiXRegressor(**kwargs)
        est.model_init()
        est.predictor.model = _LimiXModelAdapter(model)
        return est

    raise ValueError(
        f"Unknown interface_name '{interface_name}'. "
        f"Valid options: 'tabpfn_v2', 'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3', "
        f"'tabicl_v1', 'tabicl_v1_1', 'tabicl' / 'tabicl_v2', 'limix'."
    )
