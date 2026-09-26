"""Muon optimizer from TabICL.

Original code from:
    https://github.com/soda-inria/tabicl/blob/main/src/tabicl/train/_muon.py

Which was in turn adapted from:
    https://github.com/toothacher17/Megatron-LM/blob/moonshot/distributedmuon-impl/megatron/core/optimizer/muon.py

Key differences from the local Muon implementation:
- All parameters are reshaped to 2D before Newton-Schulz (handles non-2D tensors in muon groups).
- LR is scaled via ``matched_adamw_rms`` (Moonlight-style) rather than a fixed ``0.2 * sqrt(max(n,m))``.
- Single ``weight_decay`` shared by both Muon and AdamW groups (with optional cautious masking).
- Nesterov momentum is controlled by an explicit ``nesterov`` flag.
- AdamW step counter is tracked per-group rather than per-parameter.
"""

from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize G (zeroth-power approximation).

    Operates on a 2-D tensor; the caller is responsible for reshaping.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def adjust_lr_for_muon(lr: float, matched_adamw_rms: float, param_shape: tuple[int, int]) -> float:
    """Scale Muon LR to match the RMS of a reference AdamW update (Moonlight scaling).

    Parameters
    ----------
    lr:
        Base Muon learning rate.
    matched_adamw_rms:
        Target AdamW update RMS to match (0.2–0.4 recommended).
    param_shape:
        2-D shape ``(A, B)`` of the (possibly reshaped) parameter.
    """
    A, B = param_shape
    return lr * math.sqrt(max(A, B)) * matched_adamw_rms


class TabiclMuon(torch.optim.Optimizer):
    """Muon optimizer from TabICL.

    Runs Nesterov SGD momentum followed by Newton-Schulz orthogonalization.
    All parameters in ``use_muon=True`` groups are reshaped to 2D for the
    orthogonalization step (so non-matrix parameters are handled automatically).
    The effective LR is scaled by ``sqrt(max(A, B)) * matched_adamw_rms`` to
    match a reference AdamW update RMS (Moonlight scaling).

    Non-Muon groups (``use_muon=False`` or absent) are updated with AdamW.

    Parameters
    ----------
    param_groups:
        Standard PyTorch parameter groups. Set ``use_muon=True`` on groups
        to be updated with Muon; all other groups use AdamW.
    lr:
        Base learning rate for Muon. Defaults to ``0.02``.
    weight_decay:
        Weight decay applied to all groups. Defaults to ``0.1``.
    matched_adamw_rms:
        AdamW update RMS that Muon should match. Defaults to ``0.2``.
    momentum:
        Momentum coefficient for internal SGD. Defaults to ``0.95``.
    nesterov:
        Whether to use Nesterov-style momentum. Defaults to ``True``.
    ns_steps:
        Newton-Schulz iterations. Defaults to ``5``.
    adamw_betas:
        Beta coefficients for AdamW groups. Defaults to ``(0.95, 0.95)``.
    adamw_eps:
        Epsilon for AdamW groups. Defaults to ``1e-8``.
    use_cautious_wd:
        Apply weight decay only when update and parameter share the same sign
        (cautious masking). Defaults to ``False``.
    scale_wd_with_adjusted_lr:
        Apply weight decay using the Moonlight-scaled ``adjusted_lr`` instead of
        the base ``lr``.  When ``True``, WD is proportional to the actual update
        magnitude (matching AdamW semantics for each matrix shape).  When
        ``False`` (legacy behaviour), WD uses the unscaled base ``lr``, making
        it effectively weaker for large matrices.  Defaults to ``False`` for
        backward compatibility.
    """

    def __init__(
        self,
        param_groups,
        lr: float = 2e-2,
        weight_decay: float = 0.1,
        matched_adamw_rms: float = 0.2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        use_cautious_wd: bool = False,
        scale_wd_with_adjusted_lr: bool = False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            use_cautious_wd=use_cautious_wd,
            scale_wd_with_adjusted_lr=scale_wd_with_adjusted_lr,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- Muon groups ---
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            use_cautious_wd = group["use_cautious_wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            matched_adamw_rms = group["matched_adamw_rms"]
            scale_wd_with_adjusted_lr = group.get("scale_wd_with_adjusted_lr", False)

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                ns_input = g.add(buf, alpha=momentum) if nesterov else buf
                ns_input = ns_input.reshape(len(g), -1)  # reshape to 2D

                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps).view(g.shape)

                adjusted_lr = adjust_lr_for_muon(lr, matched_adamw_rms, ns_input.shape)
                wd_lr = adjusted_lr if scale_wd_with_adjusted_lr else lr

                if use_cautious_wd:
                    wd_mask = (update * p).ge(0).float()
                    p.mul_(1 - wd_lr * weight_decay * wd_mask)
                else:
                    p.mul_(1 - wd_lr * weight_decay)

                p.add_(update, alpha=-adjusted_lr)

        # --- AdamW groups ---
        for group in self.param_groups:
            if group.get("use_muon", False):
                continue

            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            use_cautious_wd = group["use_cautious_wd"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                m = state["adamw_exp_avg"]
                v = state["adamw_exp_avg_sq"]
                m.lerp_(g, 1 - beta1)
                v.lerp_(g.square(), 1 - beta2)

                update = m / (eps + v.sqrt())
                bias_correction = (1 - beta1**step) / (1 - beta2**step) ** 0.5

                if use_cautious_wd:
                    wd_mask = (update * p).ge(0).float()
                    p.mul_(1 - lr * weight_decay * wd_mask)
                else:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr / bias_correction)

        return loss
