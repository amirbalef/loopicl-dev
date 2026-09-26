"""Abstract base class for all validators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseValidator(ABC):
    """Interface that all validators must implement."""

    run_once: bool = False
    """If True, the validator is removed from the list after its first run."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as metric prefix, e.g. ``prior_val``."""
        ...

    @abstractmethod
    def validate(self, model: nn.Module, device: str, step: int | None = None, dtype: str = "float32") -> dict[str, float]:
        """Run validation and return ``{metric_name: value}`` pairs.

        The model is put back into training mode by the caller after this
        method returns, so implementations may freely call ``model.eval()``.
        """
        ...
