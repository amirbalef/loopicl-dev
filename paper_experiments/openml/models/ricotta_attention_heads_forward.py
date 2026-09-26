"""Forward-pass helpers for attention head analysis.

Provides two public functions:

forward_with_attention_head_stats
    Registers hooks on all three attention streams (ICL, within-col, cross-col)
    and captures per-head entropy, max-weight, and mean attention maps at every
    loop × block call.  Cost: 1 forward pass.

forward_with_head_ablation
    Sweeps over every (ICL block, head) pair, zeroing out that head's
    contribution and recording the per-loop accuracy drop relative to a
    clean baseline.  Cost: n_icl_blocks × n_heads + 1 forward passes
    (e.g. 3 × 8 + 1 = 25 for the default architecture).

Both functions mirror the analysis performed in
    paper/notebooks/attention_heads/attention_heads_over_loops.ipynb
    paper/notebooks/attention_heads/head_importance_ablation.ipynb
but accept pre-encoded tensors so they can be called directly from
the OpenML benchmark runner.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Attention statistics helper
# ---------------------------------------------------------------------------


def _attn_stats(
    q_BRHD: torch.Tensor,
    k_BNJD: torch.Tensor,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-head entropy, max-weight, and mean attention map.

    Parameters
    ----------
    q_BRHD : (B, R, H, d)  query
    k_BNJD : (B, N, J, d)  key — J may differ from H (GQA)

    Returns
    -------
    entropy  : np.ndarray (H,)   mean Shannon entropy per head
    max_w    : np.ndarray (H,)   mean max-weight per head
    """
    B, R, H, d = q_BRHD.shape
    N = k_BNJD.shape[1]
    J = k_BNJD.shape[2]

    if J != H:
        k_BNJD = k_BNJD.repeat_interleave(H // J, dim=2)  # (B, N, H, d)

    q_t = q_BRHD.permute(0, 2, 1, 3).float()  # (B, H, R, d)
    k_t = k_BNJD.permute(0, 2, 1, 3).float()  # (B, H, N, d)
    scores  = torch.einsum("bhrd,bhnd->bhrn", q_t, k_t) / math.sqrt(d)
    weights = scores.softmax(dim=-1)            # (B, H, R, N)

    ent     = -(weights * (weights + eps).log()).sum(dim=-1)  # (B, H, R)
    entropy = ent.mean(dim=(0, 2)).cpu().numpy()              # (H,)
    max_w   = weights.max(dim=-1).values.mean(dim=(0, 2)).cpu().numpy()  # (H,)

    return entropy, max_w


# ---------------------------------------------------------------------------
# _AttentionCapture: hooks for ICL / within-col / cross-col streams
# ---------------------------------------------------------------------------


class _AttentionCapture:
    """Register forward hooks on all three attention streams.

    Per call (= one loop step × one block) stores:
      entropy[stream]   list of np.ndarray (H,)
      max_w[stream]     list of np.ndarray (H,)
    """

    def __init__(self, ricotta_model: torch.nn.Module, num_train: int) -> None:
        self._num_train = num_train
        self._hooks: list = []
        self.entropy:  dict[str, list] = {"icl": [], "within_col": [], "cross_col": []}
        self.max_w:    dict[str, list] = {"icl": [], "within_col": [], "cross_col": []}

        emb = ricotta_model.axial_embedder

        for block in emb.icl_blocks:
            self._hooks.append(
                block.icl_attention.register_forward_hook(
                    self._make_icl_hook(), with_kwargs=True
                )
            )

        for block in emb.within_col_blocks:
            if hasattr(block, "self_attn") and hasattr(block.self_attn, "attention"):
                self._hooks.append(
                    block.self_attn.attention.register_forward_hook(
                        self._make_generic_hook("within_col")
                    )
                )

        for block in emb.cross_col_blocks:
            if hasattr(block, "attention"):
                self._hooks.append(
                    block.attention.register_forward_hook(
                        self._make_generic_hook("cross_col")
                    )
                )

    def _make_icl_hook(self):
        num_train = self._num_train

        def hook(module, args, kwargs, output):
            # Skip test-chunk calls: those pass cached_kv and only contain a
            # small chunk of test rows, so x[:, :N] would be clipped and the
            # view would fail.  Attention stats are only meaningful from the
            # train pass (full num_train rows, no cached_kv).
            if kwargs.get("cached_kv") is not None:
                return
            # single_eval_pos in the ICL attention is in token-space (may differ
            # from the sample-level num_train stored at construction time).
            N = kwargs.get("single_eval_pos", num_train)
            w_dtype = module.q_projection.weight.dtype
            x = args[0].detach().to(w_dtype)
            B, R, _ = x.shape
            H, d = module.num_heads, module.head_dim
            J = module.num_kv_heads
            with torch.no_grad():
                q = module.q_projection(x).view(B, R, H, d)
                k = module.k_projection(x[:, :N]).view(B, N, J, d)
            entropy, max_w = _attn_stats(q, k)
            self.entropy["icl"].append(entropy)
            self.max_w["icl"].append(max_w)

        return hook

    def _make_generic_hook(self, stream: str):
        def hook(module, input, output):
            w_dtype = module.q_projection.weight.dtype
            x_q = input[0].detach().to(w_dtype)
            x_k = (
                input[1].detach().to(w_dtype)
                if (len(input) > 1 and input[1] is not None)
                else x_q
            )
            B, S, _ = x_q.shape
            N = x_k.shape[1]
            H, d = module.num_heads, module.head_dim
            with torch.no_grad():
                q = module.q_projection(x_q).view(B, S, H, d)
                k = module.k_projection(x_k).view(B, N, H, d)
            entropy, max_w = _attn_stats(q, k)
            self.entropy[stream].append(entropy)
            self.max_w[stream].append(max_w)

        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# _HeadAblationHook: zeroes out one head in an ICL attention module
# ---------------------------------------------------------------------------


class _HeadAblationHook:
    """Zeroes out head ``ablate_head`` in one ICL attention module.

    Recomputes Q/K/V explicitly (bypassing Flash Attention), sets that
    head's attention weights to zero, then re-applies the output projection.
    All other heads are unaffected.
    """

    def __init__(self, module: torch.nn.Module, ablate_head: int, n_train: int) -> None:
        self.ablate_head = ablate_head
        self.n_train = n_train
        self._handle = module.register_forward_hook(self._hook_fn, with_kwargs=True)

    def _hook_fn(self, module, args, kwargs, output):
        # Test-chunk calls pass cached_kv; x only contains the test chunk, so
        # x[:, :N] would be clipped and k/v would be computed from wrong rows.
        # Skip these calls — the ablation only applies to the train pass.
        if kwargs.get("cached_kv") is not None:
            return output
        result, kv_entry = output
        N = kwargs.get("single_eval_pos", self.n_train)
        w_dtype = module.q_projection.weight.dtype
        x = args[0].detach().to(w_dtype)
        B, R, _ = x.shape
        H, d, J = module.num_heads, module.head_dim, module.num_kv_heads

        with torch.no_grad():
            q = module.q_projection(x).view(B, R, H, d).float()
            k = module.k_projection(x[:, :N]).view(B, N, J, d).float()
            v = module.v_projection(x[:, :N]).view(B, N, J, d).float()

            if J != H:
                rep = H // J
                k = k.repeat_interleave(rep, dim=2)
                v = v.repeat_interleave(rep, dim=2)

            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3)

            scores  = torch.einsum("bhrd,bhnd->bhrn", q_t, k_t) / math.sqrt(d)
            weights = scores.softmax(dim=-1)          # (B, H, R, N)
            weights[:, self.ablate_head] = 0.0        # zero out ablated head

            context = torch.einsum("bhrn,bhnd->bhrd", weights, v_t)
            context = context.permute(0, 2, 1, 3).to(w_dtype).reshape(B, R, H * d)
            new_result = module.out_projection(context)

        return (new_result.to(result.dtype), kv_entry)

    def remove(self) -> None:
        self._handle.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def forward_with_attention_head_stats(
    ricotta: torch.nn.Module,
    X_all: torch.Tensor,
    y_full: torch.Tensor,
    single_eval_pos: int,
    num_loops: int,
) -> dict:
    """Single forward pass with attention capture hooks on all streams.

    Parameters
    ----------
    ricotta          : trained LoopICL model (eval mode)
    X_all            : (1, n_train + n_test, n_features)
    y_full           : (1, n_train + n_test)  — labels for train, 0 for test
    single_eval_pos  : n_train
    num_loops        : number of ICL loops

    Returns
    -------
    dict with keys
        icl_entropy    : np.ndarray (n_loops, n_icl_blocks, n_heads)
        icl_max_w      : np.ndarray (n_loops, n_icl_blocks, n_heads)
        within_entropy : np.ndarray (n_loops, n_within_blocks, n_heads) or None
        cross_entropy  : np.ndarray (n_loops, n_cross_blocks,  n_heads) or None
        n_icl_blocks   : int
        n_icl_heads    : int
    """
    emb = ricotta.axial_embedder
    n_icl_blocks   = len(emb.icl_blocks)
    n_within_blocks = len(emb.within_col_blocks)
    n_cross_blocks  = len(emb.cross_col_blocks)
    n_icl_heads = emb.icl_blocks[0].icl_attention.num_heads

    capture = _AttentionCapture(ricotta, num_train=single_eval_pos)
    try:
        with torch.no_grad():
            ricotta(X_all, y_full, single_eval_pos=single_eval_pos, num_loops=num_loops)
    finally:
        capture.remove()

    def _reshape(data: list, n_loops: int, n_blocks: int) -> np.ndarray | None:
        if not data:
            return None
        arr = np.stack(data, axis=0)           # (n_calls, H)
        if len(data) == n_loops * n_blocks:
            return arr.reshape(n_loops, n_blocks, -1)
        return arr

    return {
        "icl_entropy":    _reshape(capture.entropy["icl"],    num_loops, n_icl_blocks),
        "icl_max_w":      _reshape(capture.max_w["icl"],      num_loops, n_icl_blocks),
        "within_entropy": _reshape(capture.entropy["within_col"], num_loops, n_within_blocks),
        "cross_entropy":  _reshape(capture.entropy["cross_col"],  num_loops, n_cross_blocks),
        "n_icl_blocks":   n_icl_blocks,
        "n_icl_heads":    n_icl_heads,
    }


def forward_with_head_ablation(
    ricotta: torch.nn.Module,
    X_all: torch.Tensor,
    y_full: torch.Tensor,
    single_eval_pos: int,
    num_loops: int,
    y_te_int: np.ndarray,
    n_classes: int,
) -> dict:
    """Ablation sweep: zeroes each (block, head) pair and records accuracy drop.

    Requires ``return_all_logits=True`` support in the LoopICL forward pass.
    Cost: n_icl_blocks × n_heads + 1 forward passes.

    Parameters
    ----------
    ricotta          : trained LoopICL model (eval mode)
    X_all            : (1, n_train + n_test, n_features)
    y_full           : (1, n_train + n_test)
    single_eval_pos  : n_train
    num_loops        : number of ICL loops
    y_te_int         : (n_test,) integer-encoded ground-truth test labels
    n_classes        : number of output classes

    Returns
    -------
    dict with keys
        baseline_accs      : np.ndarray (n_loops,)   accuracy at each loop (no ablation)
        baseline_loglosses : np.ndarray (n_loops,)   log-loss at each loop (no ablation)
        importance         : np.ndarray (n_icl_blocks, n_icl_heads, n_loops)
                             accuracy drop = baseline − ablated at each loop step
        logloss_importance : np.ndarray (n_icl_blocks, n_icl_heads, n_loops)
                             log-loss increase = ablated − baseline at each loop step
        n_icl_blocks       : int
        n_icl_heads        : int
    """
    emb = ricotta.axial_embedder
    n_icl_blocks = len(emb.icl_blocks)
    n_icl_heads  = emb.icl_blocks[0].icl_attention.num_heads

    import torch.nn.functional as F
    y_te_tensor = torch.tensor(y_te_int, dtype=torch.long)

    def _to_preds(logits_BTC: torch.Tensor) -> np.ndarray:
        return logits_BTC[0, :, :n_classes].float().cpu().argmax(dim=-1).numpy()

    def _to_probas(logits_BTC: torch.Tensor) -> np.ndarray:
        return logits_BTC[0, :, :n_classes].float().cpu().softmax(dim=-1).numpy()

    def _to_acc(logits_BTC: torch.Tensor) -> float:
        return float((_to_preds(logits_BTC) == y_te_int).mean())

    def _to_logloss(logits_BTC: torch.Tensor) -> float:
        return F.cross_entropy(
            logits_BTC[0, :, :n_classes].float().cpu(), y_te_tensor
        ).item()

    # Baseline pass
    with torch.no_grad():
        baseline_logits = ricotta(
            X_all, y_full,
            single_eval_pos=single_eval_pos,
            num_loops=num_loops,
            return_all_logits=True,
        )
    baseline_preds     = np.stack([_to_preds(lg)  for lg in baseline_logits])  # (n_loops, n_test)
    baseline_probas    = np.stack([_to_probas(lg) for lg in baseline_logits])  # (n_loops, n_test, n_classes)
    baseline_accs      = np.array([_to_acc(lg)    for lg in baseline_logits])  # (n_loops,)
    baseline_loglosses = np.array([_to_logloss(lg) for lg in baseline_logits])  # (n_loops,)

    n_test = len(y_te_int)
    importance          = np.zeros((n_icl_blocks, n_icl_heads, num_loops))
    logloss_importance  = np.zeros((n_icl_blocks, n_icl_heads, num_loops))
    ablated_preds       = np.zeros((n_icl_blocks, n_icl_heads, num_loops, n_test), dtype=baseline_preds.dtype)
    ablated_probas      = np.zeros((n_icl_blocks, n_icl_heads, num_loops, n_test, n_classes), dtype=np.float32)

    for b in range(n_icl_blocks):
        for h in range(n_icl_heads):
            hook = _HeadAblationHook(
                module      = emb.icl_blocks[b].icl_attention,
                ablate_head = h,
                n_train     = single_eval_pos,
            )
            try:
                with torch.no_grad():
                    abl_logits = ricotta(
                        X_all, y_full,
                        single_eval_pos=single_eval_pos,
                        num_loops=num_loops,
                        return_all_logits=True,
                    )
                for loop_i, logits in enumerate(abl_logits):
                    preds = _to_preds(logits)
                    ablated_preds[b, h, loop_i]      = preds
                    ablated_probas[b, h, loop_i]     = _to_probas(logits)
                    importance[b, h, loop_i]         = baseline_accs[loop_i] - float((preds == y_te_int).mean())
                    logloss_importance[b, h, loop_i] = _to_logloss(logits) - baseline_loglosses[loop_i]
            finally:
                hook.remove()

    return {
        "baseline_preds":     baseline_preds,     # (n_loops, n_test)
        "baseline_probas":    baseline_probas,    # (n_loops, n_test, n_classes)
        "baseline_accs":      baseline_accs,      # (n_loops,)
        "baseline_loglosses": baseline_loglosses, # (n_loops,)
        "ablated_preds":      ablated_preds,      # (n_blocks, n_heads, n_loops, n_test)
        "ablated_probas":     ablated_probas,     # (n_blocks, n_heads, n_loops, n_test, n_classes)
        "importance":         importance,          # (n_blocks, n_heads, n_loops)
        "logloss_importance": logloss_importance, # (n_blocks, n_heads, n_loops)
        "n_icl_blocks":       n_icl_blocks,
        "n_icl_heads":        n_icl_heads,
    }
