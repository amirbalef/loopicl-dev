"""Monkey-patches for TALENT library compatibility with newer dependency versions."""
from __future__ import annotations


def apply_sklearn_compat() -> None:
    """Translate sklearn 1.5 → 1.6 API changes in bundled TALENT models.

    sklearn 1.6 renamed ``force_all_finite`` to ``ensure_all_finite``.
    LimiX's predictor still uses the old name; patch its ``validate_data``
    method to translate the kwarg transparently so both sklearn versions work.
    """
    import sklearn as _sk

    _ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
    if _ver < (1, 6):
        return

    try:
        from TALENT.model.lib.limix.inference.predictor import LimiXPredictor

        _orig = LimiXPredictor.validate_data

        def _patched(self, x=None, y=None, reset=True, validate_separately=False, **kw):
            if "force_all_finite" in kw:
                kw["ensure_all_finite"] = kw.pop("force_all_finite")
            return _orig(self, x=x, y=y, reset=reset, validate_separately=validate_separately, **kw)

        LimiXPredictor.validate_data = _patched
    except ImportError:
        pass


def apply_realmlp_compat() -> None:
    """Patch realmlp's sklearn_base to use ``ensure_all_finite`` (sklearn >= 1.6)."""
    import sklearn as _sk

    _ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
    if _ver < (1, 6):
        return

    try:
        from sklearn.utils.validation import check_X_y as _orig_check_X_y
        from sklearn.utils.validation import check_array as _orig_check_array
        import sklearn.utils.validation as _val_mod

        def _patched_check_X_y(*args, **kw):
            if "force_all_finite" in kw:
                kw["ensure_all_finite"] = kw.pop("force_all_finite")
            return _orig_check_X_y(*args, **kw)

        def _patched_check_array(*args, **kw):
            if "force_all_finite" in kw:
                kw["ensure_all_finite"] = kw.pop("force_all_finite")
            return _orig_check_array(*args, **kw)

        _val_mod.check_X_y = _patched_check_X_y
        _val_mod.check_array = _patched_check_array

        # Also patch in realmlp's local namespace
        from TALENT.model.lib.realmlp.sklearn import sklearn_base as _rb
        _rb.check_X_y = _patched_check_X_y
        _rb.check_array = _patched_check_array
    except (ImportError, AttributeError):
        pass


def apply_limix_class_cap() -> None:
    """Raise a clear ValueError when a dataset exceeds LimiX's class limit.

    LimiX's y-encoder uses ``nn.Embedding(max_num_classes, emsize)`` fixed at
    training time.  When a dataset has more classes than that limit, the CUDA
    embedding lookup hits an out-of-bounds index and crashes with an opaque
    ``vectorized gather kernel index out of bounds`` assertion error.

    This patch intercepts ``_predict_cls`` before any GPU work and raises a
    descriptive error so the benchmark runner can skip the dataset cleanly.
    """
    try:
        import numpy as np
        from TALENT.model.lib.limix.inference.predictor import LimiXPredictor

        _orig_predict_cls = LimiXPredictor._predict_cls

        def _patched_predict_cls(self, x_train, y_train, x_test):
            n_classes = len(np.unique(y_train))
            max_model_classes = self.model.cls_y_decoder[-1].out_features
            if n_classes > max_model_classes:
                raise ValueError(
                    f"LimiX supports at most {max_model_classes} classes, "
                    f"but this dataset has {n_classes}. Skipping."
                )
            return _orig_predict_cls(self, x_train, y_train, x_test)

        LimiXPredictor._predict_cls = _patched_predict_cls
    except ImportError:
        pass


def apply_all() -> None:
    apply_sklearn_compat()
    apply_realmlp_compat()
    apply_limix_class_cap()
