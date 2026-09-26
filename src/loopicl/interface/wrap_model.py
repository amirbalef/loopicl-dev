"""Standalone transformer wrapper for all TabPFN foundation models.

Each model has a different forward calling convention.  This module provides a
unified ``wrapper(X, y_train, eval_pos)`` interface — numpy arrays in, logit
tensor out — for all supported models.

Supported models and their subclasses:
  - TabPFN v2     → TabPFNv2Wrapper
  - TabPFN v2.5   → TabPFNWrapper
  - TabICL v2     → TabICLWrapper
  - LimiX         → LimiXWrapper

Factory function (loads model internally):
    wrap_model(model_name, task_type) → TransformerWrapper

Example:
    from transformer_wrapper import wrap_model

    wrapper = wrap_model("tabpfn_v2", task_type="cls")
    logits = wrapper(X_full, y_train, eval_pos=len(y_train))
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class TransformerWrapper:
    """Base class — do not instantiate directly.

    Subclasses implement ``_forward(X_tensor, y_tensor, eval_pos)`` and store
    the underlying model in ``self._model``.  All model attributes are
    accessible through the wrapper via ``__getattr__``.
    """

    def __init__(self, model, device, dtype: str = "float32"):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_dtype", dtype)

    def __call__(
        self,
        X: np.ndarray,
        y_train: np.ndarray,
        eval_pos: int,
    ) -> torch.Tensor:
        """Run the transformer forward pass.

        Supports both single-dataset and batched-dataset inputs:

        Single  (2-D X):
            X        : ``(n_total, n_features)``
            y_train  : ``(n_train,)``
            returns  : ``(n_test, n_classes)``

        Batched (3-D X):
            X        : ``(n_datasets, n_total, n_features)``
            y_train  : ``(n_datasets, n_train)``
            returns  : ``(n_datasets, n_test, n_classes)``

        Args:
            X: Full feature matrix (train + test concatenated).
            y_train: Training labels. 1-D for single, 2-D for batched.
            eval_pos: Number of training samples (same for every dataset
               in the batch).
        """
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

    def _forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        eval_pos: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_model"), name)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        return f"{type(self).__name__}({model!r})"

# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------

class TabPFNWrapper(TransformerWrapper):
    """Wrapper for TabPFN v2.5 and v2.6.

    Forward convention:
        ``model(X, y)``
    No style argument; eval_pos is inferred internally from y_train.shape[0].
    Also used for v2.6 which shares the same interface.
    """

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        model = object.__getattribute__(self, "_model")
        if X.ndim == 3:
            # TabPFN v2.5/v2.6 only supports batch=1 internally — loop over datasets
            outputs = [
                model(X[i].unsqueeze(1), y_train[i]).squeeze(1)
                for i in range(X.shape[0])
            ]
            return torch.stack(outputs, dim=0)  # (n_datasets, n_test, n_classes)
        # single: (n_total, n_features) → (n_total, 1, n_features)
        X = X.unsqueeze(1)
        output = model(X, y_train)
        return output.squeeze(1)


# ---------------------------------------------------------------------------
# TabICL v2
# ---------------------------------------------------------------------------

class TabICLWrapper(TransformerWrapper):
    """Wrapper for TabICL v2.

    Forward convention:
        ``model(X, y_train)``
    where X has shape ``(batch, n_total, n_features)`` and y_train has shape
    ``(batch, n_train)``.  The model infers the train/test split from the length
    of y_train.

    Parameters
    ----------
    num_loops : Optional[int], default=None
        Override the number of ICL loop iterations at inference time.
        If None, uses the model's default (``icl_num_blocks``).
    loop_residual_scaling_fn : Optional[str], default=None
        Override the loop residual scaling function (e.g. ``"inv_n"``,
        ``"inv_sqrt_n"``).  If None, uses the model's trained setting.
    """

    def __init__(self, model, device, dtype: str = "float32", num_loops: int | None = None, mrl_dim: int | None = None, loop_residual_scaling_fn: str | None = None, exit_q: float | None = None):
        super().__init__(model, device, dtype)
        object.__setattr__(self, "_num_loops", num_loops)
        object.__setattr__(self, "_mrl_dim", mrl_dim)
        object.__setattr__(self, "_loop_residual_scaling_fn", loop_residual_scaling_fn)
        object.__setattr__(self, "_exit_q", exit_q)

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        model = object.__getattribute__(self, "_model")
        num_loops = object.__getattribute__(self, "_num_loops")
        mrl_dim = object.__getattribute__(self, "_mrl_dim")
        loop_residual_scaling_fn = object.__getattribute__(self, "_loop_residual_scaling_fn")
        exit_q = object.__getattribute__(self, "_exit_q")
        model.eval()
        kwargs = {"return_logits": True}
        if num_loops is not None:
            kwargs["num_loops"] = num_loops
        if mrl_dim is not None:
            kwargs["mrl_dim"] = mrl_dim
        if exit_q is not None:
            kwargs["exit_q"] = exit_q
        orig_fn = None
        if loop_residual_scaling_fn is not None and hasattr(model, "loop_residual_scaling_fn"):
            orig_fn = model.loop_residual_scaling_fn
            model.loop_residual_scaling_fn = loop_residual_scaling_fn
        try:
            if X.ndim == 3:
                output = model(X, y_train, **kwargs)
            else:
                output = model(X.unsqueeze(0), y_train.unsqueeze(0), **kwargs)
                output = output.squeeze(0)
        finally:
            if orig_fn is not None:
                model.loop_residual_scaling_fn = orig_fn
        return output


# ---------------------------------------------------------------------------
# IrisMELT two-phase wrapper
# ---------------------------------------------------------------------------

class LoopICLRegWrapper(TransformerWrapper):
    """Wrapper for LoopICLReg (bar-distribution regression).

    ``LoopICLReg.forward()`` returns ``(M, B, num_bars)`` — note the transposed
    batch/sequence axes vs the ``(B, M, n_classes)`` convention used everywhere
    else.  This wrapper corrects the layout to match the ``_TabPFNModelAdapter``
    convention (used by the TabPFN v3 Regressor interface):

    Single  (2-D X):  returns ``(n_test, num_bars)``
    Batched (3-D X):  returns ``(n_datasets, n_test, num_bars)``
    """

    def __init__(self, model, device, dtype: str = "float32", num_loops: int | None = None):
        super().__init__(model, device, dtype)
        object.__setattr__(self, "_num_loops", num_loops)

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        model = object.__getattribute__(self, "_model")
        num_loops = object.__getattribute__(self, "_num_loops")
        model.eval()
        kwargs: dict = {}
        if num_loops is not None:
            kwargs["num_loops"] = num_loops
        if X.ndim == 3:
            # batched: X (B, N_total, C), y_train (B, N_train)
            # model returns (M, B, num_bars) → permute to (B, M, num_bars)
            output = model(X, y_train, **kwargs)  # (M, B, num_bars)
            return output.permute(1, 0, 2)        # (B, M, num_bars)
        # single: X (N_total, C), y_train (N_train,) → add batch dim
        output = model(X.unsqueeze(0), y_train.unsqueeze(0), **kwargs)  # (M, 1, num_bars)
        return output[:, 0, :]  # (M, num_bars)


class IrisMELTTwoPhaseWrapper(TabICLWrapper):
    """Wrapper for IrisMELT using the two-phase ``encode_support`` / ``predict_from_cache`` path.

    Replaces the standard single-pass ``forward`` with:
      Phase 1 — ``model.encode_support(X_train, y_train)`` → ``SupportCache``
      Phase 2 — ``model.predict_from_cache(X_test, cache)`` → logits

    This eliminates the train/inference mismatch when the model was trained
    with ``forward_two_phase``.
    """

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        model = object.__getattribute__(self, "_model")
        num_loops = object.__getattribute__(self, "_num_loops")
        model.eval()
        if X.ndim == 3:
            X_train = X[:, :eval_pos]   # (B, N, C)
            X_test  = X[:, eval_pos:]   # (B, M, C)
            cache = model.encode_support(X_train, y_train, num_loops=num_loops)
            return model.predict_from_cache(X_test, cache)  # (B, M, n_classes)
        # single: add batch dim, run, remove it
        X_train = X[:eval_pos].unsqueeze(0)     # (1, N, C)
        X_test  = X[eval_pos:].unsqueeze(0)     # (1, M, C)
        cache = model.encode_support(X_train, y_train.unsqueeze(0), num_loops=num_loops)
        return model.predict_from_cache(X_test, cache).squeeze(0)  # (M, n_classes)


# ---------------------------------------------------------------------------
# LimiX
# ---------------------------------------------------------------------------

class LimiXWrapper(TransformerWrapper):
    """Wrapper for LimiX.

    Forward convention:
        ``model(x, y, eval_pos=eval_pos, task_type='cls')``
    where x has shape ``(1, n_total, n_features)`` and y has shape
    ``(1, n_total)`` with test positions zero-padded.
    """

    def __init__(self, model, device, task_type: str = "cls"):
        super().__init__(model, device)
        object.__setattr__(self, "_task_type", task_type)

    def _forward(self, X: torch.Tensor, y_train: torch.Tensor, eval_pos: int) -> torch.Tensor:
        task_type = object.__getattribute__(self, "_task_type")
        model = object.__getattribute__(self, "_model")
        if X.ndim == 3:
            # batched: (n_datasets, n_total, n_features), y_train: (n_datasets, n_train)
            n_datasets, n_total = X.shape[0], X.shape[1]
            y_full = torch.zeros(n_datasets, n_total, dtype=y_train.dtype, device=y_train.device)
            y_full[:, :eval_pos] = y_train
            output = model(X, y_full, eval_pos=eval_pos, task_type=task_type)
            return output  # (n_datasets, n_test, n_classes)
        # single
        n_total = X.shape[0]
        y_full = torch.zeros(n_total, dtype=y_train.dtype, device=y_train.device)
        y_full[:eval_pos] = y_train
        output = model(
            X.unsqueeze(0), y_full.unsqueeze(0),
            eval_pos=eval_pos, task_type=task_type,
        )
        return output.squeeze(0)  # (n_test, n_classes)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def wrap_model(
    model_name: str,
    task_type: str = "cls",
) -> TransformerWrapper:
    """Load a foundation model and return the appropriate TransformerWrapper.

    The wrapper can then be called as ``wrapper(X, y_train, eval_pos)`` with
    numpy arrays to run the transformer forward pass directly.

    Args:
        model_name: Which model to load.  One of:
                    ``"tabpfn_v2"``, ``"tabpfn_v2_5"``, ``"tabpfn_v2_6"``, ``"tabpfn_v3"``,
                    ``"tabicl_v1"``, ``"tabicl_v1_1"``, ``"tabicl"`` / ``"tabicl_v2"``,
                    ``"limix"``.
        task_type:  ``"cls"`` (classification) or ``"reg"`` (regression).

    Returns:
        The appropriate ``TransformerWrapper`` subclass instance.

    Example::

        from transformer_wrapper import wrap_model

        wrapper = wrap_model("tabpfn_v2", task_type="cls")
        logits = wrapper(X_full, y_train, eval_pos=len(y_train))

        wrapper_reg = wrap_model("tabpfn_v2", task_type="reg")
        preds = wrapper_reg(X_full, y_train, eval_pos=len(y_train))
    """
    key = model_name.lower()

    if key in ("tabpfn_v2"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = TabPFNClassifier.create_default_for_version(ModelVersion.V2) if task_type == "cls" else TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        est._initialize_model_variables()
        return TabPFNWrapper(est.model_, est.devices_[0])

    if key in ("tabpfn_v2_5", "tabpfn_2_5"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5) if task_type == "cls" else TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
        est._initialize_model_variables()
        return TabPFNWrapper(est.model_, est.devices_[0])

    if key in ("tabpfn_v2_6", "tabpfn_2_6"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = TabPFNClassifier.create_default_for_version(ModelVersion.V2_6) if task_type == "cls" else TabPFNRegressor.create_default_for_version(ModelVersion.V2_6)
        est._initialize_model_variables()
        return TabPFNWrapper(est.model_, est.devices_[0])

    if key in ("tabpfn_v3", "tabpfn_3"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        est = TabPFNClassifier.create_default_for_version(ModelVersion.V3) if task_type == "cls" else TabPFNRegressor.create_default_for_version(ModelVersion.V3)
        est._initialize_model_variables()
        return TabPFNWrapper(est.model_, est.devices_[0])

    if key in ("tabicl_v1", "tabicl_1"):
        if task_type != "cls":
            raise ValueError("TabICL v1 only supports classification.")
        from tabicl import TabICLClassifier
        est = TabICLClassifier(checkpoint_version="tabicl-classifier-v1-20250208.ckpt")
        est._resolve_device()
        est._load_model()
        est.model_.to(est.device_)
        return TabICLWrapper(est.model_, est.device_)

    if key in ("tabicl_v1_1", "tabicl_1_1"):
        if task_type != "cls":
            raise ValueError("TabICL v1.1 only supports classification.")
        from tabicl import TabICLClassifier
        est = TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-20250506.ckpt")
        est._resolve_device()
        est._load_model()
        est.model_.to(est.device_)
        return TabICLWrapper(est.model_, est.device_)

    if key in ("tabicl", "tabicl_v2", "tabicl_2"):
        from tabicl import TabICLClassifier, TabICLRegressor
        est = (
            TabICLClassifier(checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
            if task_type == "cls" else
            TabICLRegressor(checkpoint_version="tabicl-regressor-v2-20260212.ckpt")
        )
        est._resolve_device()
        est._load_model()
        est.model_.to(est.device_)
        return TabICLWrapper(est.model_, est.device_)

    if key == "limix":
        from limix import LimiXClassifier, LimiXRegressor
        est = LimiXClassifier() if task_type == "cls" else LimiXRegressor()
        est.model_init()
        device = next(est.predictor.model.parameters()).device
        return LimiXWrapper(est.predictor.model, device, task_type=task_type)

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Valid options: 'tabpfn_v2', 'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3', "
        f"'tabicl_v1', 'tabicl_v1_1', 'tabicl' / 'tabicl_v2', 'limix'."
    )
