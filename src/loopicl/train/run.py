"""Entry point for TabICL training."""

from __future__ import annotations

from importlib import import_module

from torch.multiprocessing import set_start_method

from loopicl.train.trainer import Trainer
from loopicl.models.utils import load_model_config


def _load_trainer_class(config) -> type:
    """Return the trainer class specified in config.general, defaulting to Trainer."""
    trainer_module = getattr(config.general, "trainer_module", None)
    trainer_class  = getattr(config.general, "trainer_class",  None)
    if trainer_module and trainer_class:
        return getattr(import_module(trainer_module), trainer_class)
    return Trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to train")
    args = parser.parse_args()

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Context already set

    config = load_model_config(args.model_name)
    TrainerClass = _load_trainer_class(config)
    trainer = TrainerClass(model_name=args.model_name)
    trainer.train()
