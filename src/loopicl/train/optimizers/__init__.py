from __future__ import annotations

import math
from functools import partial

from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from .tabicl_muon import TabiclMuon


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_end: float = 0.0,
    lr_init: float = 1.0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return lr_end / lr_init
    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = amplitude_decay**current_cycle
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init


def get_cosine_with_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    amplitude_decay: float = 1.0,
    lr_end: float = 0.0,
    last_epoch: int = -1,
):
    lr_lambdas = []
    for group in optimizer.param_groups:
        lr_init = group["lr"]
        if lr_end > lr_init:
            raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init}) for param group")
        lr_lambdas.append(partial(
            _get_cosine_with_restarts_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            amplitude_decay=amplitude_decay,
            lr_end=lr_end,
            lr_init=lr_init,
        ))
    return LambdaLR(optimizer, lr_lambdas, last_epoch)


def _get_wsd_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_training_steps: int,
    lr_end: float,
    lr_init: float,
    decay_type: str = "cosine",
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_stable_steps:
        return 1.0
    decay_steps = num_training_steps - num_stable_steps
    progress = float(current_step - num_stable_steps) / float(max(1, decay_steps))
    progress = min(progress, 1.0)
    min_factor = lr_end / max(lr_init, 1e-12)
    if decay_type == "linear":
        return 1.0 - progress * (1.0 - min_factor)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_wsd_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_fraction: float = 0.1,
    lr_end: float = 0.0,
    decay_type: str = "cosine",
    last_epoch: int = -1,
):
    num_decay_steps = int(num_training_steps * decay_fraction)
    num_stable_steps = num_training_steps - num_decay_steps
    lr_lambdas = []
    for group in optimizer.param_groups:
        lr_init = group["lr"]
        lr_lambdas.append(partial(
            _get_wsd_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            lr_init=lr_init,
            decay_type=decay_type,
        ))
    return LambdaLR(optimizer, lr_lambdas, last_epoch)


def get_scheduler(config, optimizer):
    """Get the learning rate scheduler based on the training configuration."""
    if config.warmup_proportion is not None:
        warmup_steps = config.max_steps * config.warmup_proportion
    else:
        warmup_steps = config.warmup_steps

    if config.scheduler == "constant":
        return get_constant_schedule(optimizer=optimizer)
    elif config.scheduler == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.max_steps
        )
    elif config.scheduler == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.max_steps
        )
    elif config.scheduler == "cosine_with_restarts":
        return get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            num_cycles=config.cosine_num_cycles,
            amplitude_decay=config.cosine_amplitude_decay,
            lr_end=config.cosine_lr_end,
        )
    elif config.scheduler == "wsd":
        return get_wsd_scheduler(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            decay_fraction=config.wsd_decay_fraction,
            lr_end=config.wsd_lr_end,
            decay_type=config.wsd_decay_type,
        )
    elif config.scheduler == "polynomial_decay_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            lr_end=config.poly_decay_lr_end,
            power=config.poly_decay_power,
        )
    else:
        raise NotImplementedError(f"Scheduler '{config.scheduler}' is not supported.")


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def get_optimizer(config, params):
    """Instantiate an optimizer based on ``config.optimizer`` (default: ``adamw``)."""
    name = getattr(config, "optimizer", "adamw").lower()
    lr = config.lr
    wd = config.weight_decay

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        momentum = getattr(config, "sgd_momentum", 0.9)
        nesterov = getattr(config, "sgd_nesterov", False)
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    elif name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=wd)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError:
            raise ImportError("Install lion-pytorch to use the Lion optimizer: pip install lion-pytorch")
        return Lion(params, lr=lr, weight_decay=wd)
    elif name == "tabicl_muon":
        params = list(params)
        muon_lr = getattr(config, "muon_lr", lr)
        muon_momentum = getattr(config, "muon_momentum", 0.95)
        muon_ns_steps = getattr(config, "muon_ns_steps", 5)
        matched_adamw_rms = getattr(config, "muon_matched_adamw_rms", 0.2)
        use_cautious_wd = getattr(config, "muon_cautious_wd", False)
        scale_wd_with_adjusted_lr = getattr(config, "muon_scale_wd_with_adjusted_lr", False)
        adamw_lr = getattr(config, "muon_adamw_lr", lr)
        muon_params = [p for p in params if p.ndim >= 2 and min(p.shape) >= 64]
        adamw_params = [p for p in params if p.ndim < 2 or min(p.shape) < 64]
        param_groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adamw_params, "use_muon": False, "lr": adamw_lr},
        ]
        adamw_betas = getattr(config, "muon_adamw_betas", (0.95, 0.95))
        return TabiclMuon(
            param_groups=param_groups,
            lr=muon_lr,
            weight_decay=wd,
            matched_adamw_rms=matched_adamw_rms,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            adamw_betas=adamw_betas,
            use_cautious_wd=use_cautious_wd,
            scale_wd_with_adjusted_lr=scale_wd_with_adjusted_lr,
        )
    else:
        raise NotImplementedError(f"Optimizer '{name}' is not supported.")


__all__ = ["get_optimizer", "get_scheduler", "TabiclMuon"]
